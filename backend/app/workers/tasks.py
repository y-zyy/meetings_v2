"""Celery tasks for the staged ASR + LLM processing pipeline."""

import logging
import os
from datetime import datetime, timezone

from celery import chain
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Sync engine for Celery workers (Celery does not use asyncio)
_sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
_engine = create_engine(_sync_url, pool_size=1, max_overflow=1, pool_pre_ping=True)
SyncSession = sessionmaker(bind=_engine)


def _get_meeting(session: Session, meeting_id: int):
    from app.models.meeting import Meeting
    return session.get(Meeting, meeting_id)


def _save_markdown_snapshot(meeting, suffix: str, heading: str, body: str) -> None:
    """오디오 파일과 같은 디렉터리에 처리 단계별 결과를 markdown 파일로 저장.

    파일명: {uid}_{suffix}.md (uid는 오디오 파일명에서 확장자를 뗀 부분)
    저장 실패는 전체 파이프라인을 막지 않도록 로그만 남기고 무시한다.
    """
    if not meeting.file_path:
        return
    try:
        dest_dir = os.path.dirname(meeting.file_path)
        uid, _ = os.path.splitext(os.path.basename(meeting.file_path))
        md_path = os.path.join(dest_dir, f"{uid}_{suffix}.md")

        generated_at = datetime.now(timezone.utc).isoformat()
        header = (
            f"# {heading}\n\n"
            f"- 회의: {meeting.title}\n"
            f"- 회의 ID: {meeting.id}\n"
            f"- 생성 일시(UTC): {generated_at}\n\n"
            "---\n\n"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(header + (body or ""))
        logger.info("[%s] markdown 저장 완료: %s", meeting.id, md_path)
    except Exception:
        logger.exception("[%s] markdown 저장 실패 (suffix=%s)", meeting.id, suffix)


def _retry_stage(task, session: Session, meeting_id: int, stage: str, exc: Exception):
    """Retry only the failed stage and expose failure after retries are exhausted."""
    session.rollback()
    logger.exception("[%s] %s 단계 실패: %s", meeting_id, stage, exc)

    if task.request.retries >= task.max_retries:
        try:
            meeting = _get_meeting(session, meeting_id)
            if meeting:
                meeting.status = "failed"
                meeting.error_message = f"{stage}: {exc}"[:500]
                session.commit()
        except Exception:
            logger.exception("[%s] 실패 상태 저장 실패", meeting_id)
            session.rollback()
        raise exc

    countdown = min(60, 10 * (2 ** task.request.retries))
    raise task.retry(exc=exc, countdown=countdown)


@celery_app.task(bind=True, name="process_meeting")
def process_meeting(self, meeting_id: int):
    """Replace the compatibility entry point with a checkpointed stage chain."""
    workflow = chain(
        transcribe_meeting.si(meeting_id),
        postprocess_meeting.si(meeting_id),
        generate_minutes_task.si(meeting_id),
    )
    return self.replace(workflow)


@celery_app.task(bind=True, name="process_meeting.asr", max_retries=2)
def transcribe_meeting(self, meeting_id: int):
    from app.models.user import User  # noqa: F401 — registers relationship mapper
    from app.services import asr
    from app.services.runtime_settings import get_effective_settings_sync

    with SyncSession() as session:
        meeting = _get_meeting(session, meeting_id)
        if not meeting:
            logger.error("Meeting %s not found", meeting_id)
            return meeting_id

        try:
            meeting.status = "asr_processing"
            meeting.error_message = None
            session.commit()
            logger.info("[%s] ASR 시작: %s", meeting_id, meeting.file_path)

            effective = get_effective_settings_sync(session)
            transcript = asr.transcribe(meeting.file_path, effective)
            meeting.transcript = transcript
            session.commit()

            logger.info("[%s] ASR 완료 (%d chars)", meeting_id, len(transcript))
            _save_markdown_snapshot(meeting, "01_asr", "음성인식 결과 (ASR)", transcript)
            return meeting_id
        except Exception as exc:
            _retry_stage(self, session, meeting_id, "ASR", exc)


@celery_app.task(bind=True, name="process_meeting.postprocess", max_retries=2)
def postprocess_meeting(self, meeting_id: int):
    from sqlalchemy import select as sa_select

    from app.models.glossary import AdminCorrectionRule, AdminGlossaryTerm, UserGlossaryTerm
    from app.models.user import User  # noqa: F401 — registers relationship mapper
    from app.services import asr_postprocess
    from app.services.asr_postprocess_rule import apply_rule_based_correction, load_seed_pairs
    from app.services.runtime_settings import get_effective_settings_sync
    from app.services.sentence_split import split_into_lines

    with SyncSession() as session:
        meeting = _get_meeting(session, meeting_id)
        if not meeting:
            logger.error("Meeting %s not found", meeting_id)
            return meeting_id

        try:
            meeting.status = "stt_postprocessing"
            session.commit()
            logger.info("[%s] STT 후처리 시작", meeting_id)

            effective = get_effective_settings_sync(session)
            transcript = meeting.transcript or ""

            seed_pairs = load_seed_pairs(settings.CORRECTION_RULES_SEED_PATH)
            correction_rules = session.execute(
                sa_select(AdminCorrectionRule).order_by(AdminCorrectionRule.created_at)
            ).scalars().all()
            db_pairs = [(r.wrong, r.correct) for r in correction_rules]
            rule_pairs = list({**dict(seed_pairs), **dict(db_pairs)}.items())
            if rule_pairs:
                transcript = apply_rule_based_correction(transcript, rule_pairs)
                logger.info(
                    "[%s] Rule-based 교정 완료 (시드 %d + DB %d = 총 %d 규칙)",
                    meeting_id, len(seed_pairs), len(db_pairs), len(rule_pairs),
                )

            admin_terms = [
                r.term + (f" ({r.description})" if r.description else "")
                for r in session.execute(
                    sa_select(AdminGlossaryTerm).order_by(AdminGlossaryTerm.created_at)
                ).scalars().all()
            ]
            user_terms = [
                r.term + (f" ({r.description})" if r.description else "")
                for r in session.execute(
                    sa_select(UserGlossaryTerm)
                    .where(UserGlossaryTerm.user_id == meeting.created_by)
                    .order_by(UserGlossaryTerm.order, UserGlossaryTerm.created_at)
                ).scalars().all()
            ]

            transcript = asr_postprocess.postprocess_transcript(
                transcript=transcript,
                title=meeting.title,
                meeting_date=str(meeting.meeting_date) if meeting.meeting_date else "",
                location=meeting.location,
                attendees=meeting.attendees,
                agenda=meeting.agenda,
                notes=meeting.notes,
                admin_terms=admin_terms,
                user_terms=user_terms,
                effective=effective,
            )
            transcript = split_into_lines(transcript)

            meeting.transcript = transcript
            session.commit()
            logger.info("[%s] STT 후처리 완료 (%d chars)", meeting_id, len(transcript))
            _save_markdown_snapshot(
                meeting, "02_asr_postprocessed", "음성인식 후처리 결과", transcript
            )
            return meeting_id
        except Exception as exc:
            _retry_stage(self, session, meeting_id, "STT 후처리", exc)


@celery_app.task(bind=True, name="process_meeting.minutes", max_retries=2)
def generate_minutes_task(self, meeting_id: int):
    from app.models.meeting import ActionItem, Decision
    from app.models.user import User  # noqa: F401 — registers relationship mapper
    from app.services import llm
    from app.services.runtime_settings import get_effective_settings_sync

    with SyncSession() as session:
        meeting = _get_meeting(session, meeting_id)
        if not meeting:
            logger.error("Meeting %s not found", meeting_id)
            return meeting_id

        try:
            meeting.status = "llm_processing"
            session.commit()
            logger.info("[%s] LLM 시작", meeting_id)

            effective = get_effective_settings_sync(session)
            transcript = meeting.transcript or ""
            result = llm.generate_minutes(
                title=meeting.title,
                meeting_date=str(meeting.meeting_date) if meeting.meeting_date else "",
                location=meeting.location,
                attendees=meeting.attendees,
                agenda=meeting.agenda,
                transcript=transcript,
                effective=effective,
            )

            meeting.summary = result.get("summary", "")

            session.query(Decision).filter_by(meeting_id=meeting_id).delete()
            for i, content in enumerate(result.get("decisions", [])):
                session.add(Decision(meeting_id=meeting_id, content=str(content), order=i))

            session.query(ActionItem).filter_by(meeting_id=meeting_id).delete()
            for i, item in enumerate(result.get("action_items", [])):
                session.add(ActionItem(
                    meeting_id=meeting_id,
                    content=str(item.get("content", "")),
                    assignee=item.get("assignee") or None,
                    due_date=_parse_date(item.get("due_date")),
                    status="pending",
                    order=i,
                ))

            meeting.status = "done"
            session.commit()
            _save_markdown_snapshot(meeting, "03_summary", "LLM 요약 결과", meeting.summary)
            logger.info("[%s] 처리 완료", meeting_id)
            return meeting_id
        except Exception as exc:
            _retry_stage(self, session, meeting_id, "회의록 생성", exc)


def _parse_date(value):
    if not value:
        return None
    from datetime import date
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None

"""Celery tasks for ASR + LLM processing pipeline."""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Sync engine for Celery workers (Celery does not use asyncio)
_sync_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
_engine = create_engine(_sync_url, pool_size=5, max_overflow=10, pool_pre_ping=True)
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


def _mark_failed(session: Session, meeting_id: int, exc: Exception) -> None:
    logger.exception("[%s] 처리 실패: %s", meeting_id, exc)
    session.rollback()
    try:
        meeting = _get_meeting(session, meeting_id)
        if meeting:
            meeting.status = "failed"
            meeting.error_message = str(exc)[:500]
            session.commit()
    except Exception:
        pass


# ASR과 LLM은 서로 다른 성격의 작업(각각 별도 서버로의 네트워크 호출)이라
# 별도 큐(asr / llm)로 분리해서 체이닝한다. 큐마다 별도 워커 프로세스를
# 띄우므로(infra/docker-compose.yml), 한쪽이 몰려도 다른 쪽 처리량에
# 영향을 주지 않는다. 자세한 배경은 커밋 메시지 참고.
@celery_app.task(
    bind=True, name="process_meeting_asr", max_retries=2,
    time_limit=settings.ASR_TIMEOUT + 60, soft_time_limit=settings.ASR_TIMEOUT,
)
def process_meeting_asr(self, meeting_id: int):
    from app.models.user import User  # noqa: F401 — registers User mapper for Meeting.owner relationship
    from app.models.meeting import Meeting
    from app.services import asr
    from app.services.runtime_settings import get_effective_settings_sync

    with SyncSession() as session:
        meeting = _get_meeting(session, meeting_id)
        if not meeting:
            logger.error("Meeting %s not found", meeting_id)
            return

        # Read effective settings (DB overrides > env defaults) once per job
        effective = get_effective_settings_sync(session)

        try:
            # ── Step 1: ASR ───────────────────────────────────────────────
            meeting.status = "asr_processing"
            session.commit()
            logger.info("[%s] ASR 시작: %s", meeting_id, meeting.file_path)

            transcript = asr.transcribe(meeting.file_path, effective)
            meeting.transcript = transcript
            session.commit()
            logger.info("[%s] ASR 완료 (%d chars)", meeting_id, len(transcript))
            _save_markdown_snapshot(meeting, "01_asr", "음성인식 결과 (ASR)", transcript)

        except Exception as exc:
            _mark_failed(session, meeting_id, exc)
            raise self.retry(exc=exc, countdown=10) from exc

    # LLM 큐로 다음 단계를 넘긴다 (DB 세션 종료 후 큐잉).
    process_meeting_llm.delay(meeting_id)


@celery_app.task(
    bind=True, name="process_meeting_llm", max_retries=2,
    # 이 태스크 안에서 LLM을 최대 3번(STT 후처리 1회 + 회의록 초안/정제 2회)
    # 순차 호출하므로 LLM_TIMEOUT의 3배를 상한으로 잡는다.
    time_limit=settings.LLM_TIMEOUT * 3 + 60, soft_time_limit=settings.LLM_TIMEOUT * 3,
)
def process_meeting_llm(self, meeting_id: int):
    from app.models.user import User  # noqa: F401 — registers User mapper for Meeting.owner relationship
    from app.models.meeting import ActionItem, Decision, Meeting
    from app.models.glossary import AdminCorrectionRule, AdminGlossaryTerm, UserGlossaryTerm
    from app.services import asr_postprocess, llm
    from app.services.asr_postprocess_rule import apply_rule_based_correction, load_seed_pairs
    from app.services.runtime_settings import get_effective_settings_sync
    from app.services.sentence_split import split_into_lines
    from sqlalchemy import select as sa_select

    with SyncSession() as session:
        meeting = _get_meeting(session, meeting_id)
        if not meeting:
            logger.error("Meeting %s not found", meeting_id)
            return

        # Read effective settings (DB overrides > env defaults) once per job
        effective = get_effective_settings_sync(session)
        transcript = meeting.transcript or ""

        try:
            # ── Step 1.5: STT 후처리 ──────────────────────────────────────
            meeting.status = "stt_postprocessing"
            session.commit()
            logger.info("[%s] STT 후처리 시작", meeting_id)

            # ── Step 1.5a: Rule-based 교정 (Aho-Corasick) ────────────────
            # 시드 파일(대량 사전) + DB(관리자 UI 개별 등록) 병합, 동일 wrong은 DB가 우선
            seed_pairs = load_seed_pairs(settings.CORRECTION_RULES_SEED_PATH)
            correction_rules = session.execute(
                sa_select(AdminCorrectionRule).order_by(AdminCorrectionRule.created_at)
            ).scalars().all()
            db_pairs = [(r.wrong, r.correct) for r in correction_rules]
            rule_pairs = list({**dict(seed_pairs), **dict(db_pairs)}.items())
            if rule_pairs:
                transcript = apply_rule_based_correction(transcript, rule_pairs)
                meeting.transcript = transcript
                session.commit()
                logger.info(
                    "[%s] Rule-based 교정 완료 (시드 %d + DB %d = 총 %d 규칙)",
                    meeting_id, len(seed_pairs), len(db_pairs), len(rule_pairs),
                )

            # ── Step 1.5b: LLM 기반 STT 후처리 ──────────────────────────
            admin_terms = [
                (r.term + (f" ({r.description})" if r.description else ""))
                for r in session.execute(sa_select(AdminGlossaryTerm).order_by(AdminGlossaryTerm.created_at)).scalars().all()
            ]
            user_terms = [
                (r.term + (f" ({r.description})" if r.description else ""))
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

            # ── Step 1.5c: 문장 단위 줄바꿈 포맷팅 (Kiwi) ────────────────
            transcript = split_into_lines(transcript)

            meeting.transcript = transcript
            session.commit()
            logger.info("[%s] STT 후처리 완료 (%d chars)", meeting_id, len(transcript))
            _save_markdown_snapshot(meeting, "02_asr_postprocessed", "음성인식 후처리 결과", transcript)

            # ── Step 2: LLM ───────────────────────────────────────────────
            meeting.status = "llm_processing"
            session.commit()
            logger.info("[%s] LLM 시작", meeting_id)

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

            # Replace existing decisions / action items
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

            _save_markdown_snapshot(meeting, "03_summary", "LLM 요약 결과", meeting.summary)

            meeting.status = "done"
            session.commit()
            logger.info("[%s] 처리 완료", meeting_id)

        except Exception as exc:
            _mark_failed(session, meeting_id, exc)
            raise self.retry(exc=exc, countdown=10) from exc


def _parse_date(value):
    if not value:
        return None
    from datetime import date
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None

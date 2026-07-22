"""Celery tasks for ASR + LLM processing pipeline."""

import logging

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


@celery_app.task(bind=True, name="process_meeting", max_retries=2)
def process_meeting(self, meeting_id: int):
    from app.models.user import User  # noqa: F401 — registers User mapper for Meeting.owner relationship
    from app.models.meeting import ActionItem, Decision, Meeting
    from app.services import asr, llm
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

            # ── Step 1.5: STT 후처리 ──────────────────────────────────────
            meeting.status = "stt_postprocessing"
            session.commit()
            logger.info("[%s] STT 후처리 시작", meeting_id)

            from app.models.glossary import AdminCorrectionRule, AdminGlossaryTerm, UserGlossaryTerm
            from app.services import asr_postprocess
            from app.services.asr_postprocess_rule import apply_rule_based_correction
            from app.services.sentence_split import split_into_lines
            from sqlalchemy import select as sa_select

            # ── Step 1.5a: Rule-based 교정 (Aho-Corasick) ────────────────
            correction_rules = session.execute(
                sa_select(AdminCorrectionRule).order_by(AdminCorrectionRule.created_at)
            ).scalars().all()
            rule_pairs = [(r.wrong, r.correct) for r in correction_rules]
            if rule_pairs:
                transcript = apply_rule_based_correction(transcript, rule_pairs)
                meeting.transcript = transcript
                session.commit()
                logger.info("[%s] Rule-based 교정 완료 (%d 규칙)", meeting_id, len(rule_pairs))

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

            meeting.status = "done"
            session.commit()
            logger.info("[%s] 처리 완료", meeting_id)

        except Exception as exc:
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
            raise self.retry(exc=exc, countdown=10) from exc


def _parse_date(value):
    if not value:
        return None
    from datetime import date
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None

from celery import Celery

from app.config import settings


_soft_limit = settings.CELERY_TASK_SOFT_TIME_LIMIT
_hard_limit = max(settings.CELERY_TASK_TIME_LIMIT, _soft_limit + 60)
_visibility_timeout = max(settings.CELERY_VISIBILITY_TIMEOUT, _hard_limit + 60)

celery_app = Celery(
    "meetings",
    broker=settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_ignore_result=True,
    task_store_errors_even_if_ignored=False,
    task_soft_time_limit=_soft_limit,
    task_time_limit=_hard_limit,
    broker_transport_options={"visibility_timeout": _visibility_timeout},
    visibility_timeout=_visibility_timeout,
    task_routes={
        "process_meeting": {"queue": "ingest"},
        "process_meeting.ingest": {"queue": "ingest"},
        "process_meeting.asr": {"queue": "asr"},
        "process_meeting.postprocess": {"queue": "postprocess"},
        "process_meeting.diarize": {"queue": "asr"},
        "process_meeting.minutes": {"queue": "minutes"},
    },
)


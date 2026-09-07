from celery import Celery

from app.config import settings

celery_app = Celery(
    "meetings",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL.replace("/0", "/1"),
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker slot
    # ASR (network call to a remote/GPU ASR server) and LLM (network calls to
    # the LLM server) run on separate queues so a run of LLM-heavy meetings
    # can't starve ASR workers, and vice versa. Each queue's worker process is
    # started with its own --queues/--concurrency in docker-compose - see
    # infra/docker-compose.yml. Per-task time limits (set on the @task
    # decorators in app/workers/tasks.py) replace the old combined
    # ASR+LLM limit now that the two stages are separate tasks.
    task_routes={
        "process_meeting_asr": {"queue": "asr"},
        "process_meeting_llm": {"queue": "llm"},
    },
)

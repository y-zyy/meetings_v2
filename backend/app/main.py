import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import AsyncSessionLocal, engine
from app.routers import auth, meetings, admin
from app.routers import settings_admin
from app.routers import board
from app.routers import glossary

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="KAI Meetings API", version="1.0.0", docs_url="/api/docs", redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(meetings.router)
app.include_router(admin.router)
app.include_router(settings_admin.router)
app.include_router(board.router)
app.include_router(glossary.router)


@app.on_event("startup")
async def on_startup():
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Create tables & seed admin user
    from app.database import Base
    from app.models import user, meeting, setting, board, glossary  # register models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await _seed_admin()


async def _seed_admin():
    from sqlalchemy import select
    from app.models.user import User
    from app.core.security import hash_password

    log = logging.getLogger(__name__)
    async with AsyncSessionLocal() as db:
        existing = (await db.execute(select(User).where(User.username == settings.ADMIN_USERNAME))).scalar_one_or_none()
        if not existing:
            admin = User(
                username=settings.ADMIN_USERNAME,
                email=settings.ADMIN_EMAIL,
                full_name="관리자",
                hashed_password=hash_password(settings.ADMIN_PASSWORD),
                role="admin",
            )
            db.add(admin)
            await db.commit()
            log.info("Admin user '%s' created.", settings.ADMIN_USERNAME)
        else:
            # Restore admin account if it was deactivated or its role was changed
            changed = False
            if not existing.is_active:
                existing.is_active = True
                changed = True
                log.warning("Admin user '%s' was inactive — re-activating.", settings.ADMIN_USERNAME)
            if existing.role != "admin":
                existing.role = "admin"
                changed = True
                log.warning("Admin user '%s' had wrong role — restoring to 'admin'.", settings.ADMIN_USERNAME)
            if changed:
                await db.commit()


@app.get("/api/health")
async def health():
    return {"status": "ok"}

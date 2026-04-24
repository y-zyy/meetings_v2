import io
import urllib.parse
import zipfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from database import get_db
from generate import build_document, build_transcript_document
from models import Meeting, User
from schemas import MeetingDetail, MeetingListItem, MeetingUpdate

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.get("", response_model=list[MeetingListItem])
async def list_meetings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Meeting)
        .where(Meeting.user_id == current_user.id)
        .order_by(Meeting.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{meeting_id}", response_model=MeetingDetail)
async def get_meeting(
    meeting_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == current_user.id)
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="회의를 찾을 수 없습니다.")
    return meeting


@router.patch("/{meeting_id}", response_model=MeetingDetail)
async def update_meeting(
    meeting_id: int,
    body: MeetingUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == current_user.id)
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="회의를 찾을 수 없습니다.")

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(meeting, field, value)

    await db.commit()
    await db.refresh(meeting)
    return meeting


@router.delete("/{meeting_id}", status_code=204)
async def delete_meeting(
    meeting_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == current_user.id)
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="회의를 찾을 수 없습니다.")

    await db.delete(meeting)
    await db.commit()


@router.get("/{meeting_id}/download")
async def download_meeting(
    meeting_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Meeting).where(Meeting.id == meeting_id, Meeting.user_id == current_user.id)
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail="회의를 찾을 수 없습니다.")
    if meeting.status != "completed" or not meeting.meeting_json:
        raise HTTPException(status_code=409, detail="처리가 완료되지 않은 회의입니다.")

    metadata = {
        "회의명": meeting.title,
        "일시": meeting.date_time or "",
        "장소": meeting.location or "",
        "참석자": meeting.participants or "",
        "안건": meeting.agenda or [],
    }

    minutes_buf = io.BytesIO()
    build_document(meeting.meeting_json).save(minutes_buf)
    minutes_buf.seek(0)

    transcript_buf = io.BytesIO()
    build_transcript_document(meeting.refined_transcript or "", metadata).save(transcript_buf)
    transcript_buf.seek(0)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("minutes.docx", minutes_buf.read())
        zf.writestr("transcript.docx", transcript_buf.read())
    zip_buf.seek(0)

    zip_filename = f"{meeting.title or '회의'}_결과.zip"
    encoded_filename = urllib.parse.quote(zip_filename)

    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="result.zip"; '
                f"filename*=UTF-8''{encoded_filename}"
            )
        },
    )

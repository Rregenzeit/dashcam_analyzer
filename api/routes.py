from __future__ import annotations
import re
import shutil
from pathlib import Path
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from .job_manager import job_manager
from .worker import submit_job

router = APIRouter()
UPLOADS_DIR = Path("uploads")


@router.post("/upload", status_code=202)
async def upload_video(file: UploadFile = File(...)):
    job_id = job_manager.create_job("")
    upload_path = UPLOADS_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)
    video_path = upload_path / "original.mp4"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    submit_job(job_id, str(video_path))
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    info = job_manager.get_job(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return info


@router.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    info = job_manager.get_job(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if info.status != "done":
        raise HTTPException(status_code=202, detail=f"Job not complete: {info.status}")
    result = job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Result file missing")
    return result


@router.get("/jobs/{job_id}/clips")
async def list_clips(job_id: str):
    info = job_manager.get_job(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    clips = job_manager.list_clips(job_id)
    return {"clips": [f"/jobs/{job_id}/clips/{c}" for c in clips]}


@router.get("/jobs/{job_id}/clips/{filename}")
async def download_clip(job_id: str, filename: str, request: Request):
    clip_path = Path("outputs") / job_id / "clips" / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    file_size = clip_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            end = min(end, file_size - 1)
            chunk = end - start + 1

            def _iter():
                with clip_path.open("rb") as f:
                    f.seek(start)
                    remaining = chunk
                    while remaining > 0:
                        data = f.read(min(65536, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(
                _iter(),
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk),
                    "Content-Disposition": f'inline; filename="{filename}"',
                },
            )

    return FileResponse(
        str(clip_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )

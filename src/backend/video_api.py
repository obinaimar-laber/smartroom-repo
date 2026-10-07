"""
Video Translate API
Endpoints untuk upload, status, download, dan delete video translate jobs.

Dipanggil dari main.py dengan:
    from video_api import router as video_router
    app.include_router(video_router)
"""

import os
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import PlainTextResponse

from pipeline import LANG_MAP
from video_processor import (
    create_job,
    get_job,
    update_job,
    list_jobs,
    delete_job,
    check_expired_jobs,
    get_output_path,
    process_video,
    UPLOAD_DIR,
    MAX_FILE_SIZE_MB,
)


router = APIRouter(prefix="/api/video", tags=["video"])


# ============================================================
# VALID VIDEO EXTENSIONS
# ============================================================
VALID_VIDEO_EXTS = ["mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "3gp"]


# ============================================================
# POST /api/video/upload
# ============================================================
@router.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    source_lang: str = Form("ru"),
    target_lang: str = Form("id"),
):
    """
    Upload video untuk di-translate.
    Return job_id yang bisa dipakai untuk cek status.
    """
    # Auto-cleanup expired jobs
    check_expired_jobs()

    # Validasi bahasa
    if source_lang not in LANG_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid source_lang: {source_lang}")
    if target_lang not in LANG_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid target_lang: {target_lang}")

    # Cek ukuran file (dengan seek)
    video.file.seek(0, os.SEEK_END)
    file_size = video.file.tell()
    video.file.seek(0)
    file_size_mb = file_size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)",
        )

    # Buat job
    job_id = create_job(
        filename=video.filename or "unknown",
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # Tentukan ekstensi
    ext = "mp4"
    if video.filename and "." in video.filename:
        detected_ext = video.filename.rsplit(".", 1)[-1].lower()
        if detected_ext in VALID_VIDEO_EXTS:
            ext = detected_ext

    video_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")

    # Simpan file (streaming 1 MB per chunk, hemat RAM)
    try:
        with open(video_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        delete_job(job_id)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Update job
    update_job(
        job_id,
        status="pending",
        progress=0,
        message=f"File uploaded ({file_size_mb:.1f} MB). Starting...",
    )

    # Jalankan processing di background
    asyncio.create_task(
        process_video(
            job_id=job_id,
            video_path=video_path,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    )

    print(f"[Video API] Job {job_id} started: {video.filename} ({file_size_mb:.1f} MB)")

    return {
        "job_id": job_id,
        "status": "pending",
        "filename": video.filename,
        "file_size_mb": round(file_size_mb, 1),
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


# ============================================================
# GET /api/video/status/{job_id}
# ============================================================
@router.get("/status/{job_id}")
async def video_status(job_id: str):
    """Cek status job video."""
    check_expired_jobs()

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "filename": job["filename"],
        "source_lang": job["source_lang"],
        "target_lang": job["target_lang"],
        "duration": job.get("duration", 0),
        "duration_min": round(job.get("duration", 0) / 60, 1) if job.get("duration") else 0,
        "total_chunks": job.get("total_chunks", 0),
        "current_chunk": job.get("current_chunk", 0),
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
    }


# ============================================================
# GET /api/video/jobs
# ============================================================
@router.get("/jobs")
async def video_list_jobs():
    """List semua job video (untuk UI)."""
    check_expired_jobs()

    jobs = list_jobs()
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "progress": j["progress"],
                "message": j["message"],
                "filename": j["filename"],
                "source_lang": j["source_lang"],
                "target_lang": j["target_lang"],
                "duration_min": round(j.get("duration", 0) / 60, 1) if j.get("duration") else 0,
                "created_at": j["created_at"],
                "completed_at": j.get("completed_at"),
            }
            for j in jobs
        ]
    }


# ============================================================
# GET /api/video/download/{job_id}?format=txt|json
# ============================================================
@router.get("/download/{job_id}")
async def video_download(job_id: str, format: str = "txt"):
    """
    Download hasil terjemahan video.
    format: txt atau json
    """
    check_expired_jobs()

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Job not done yet. Status: {job['status']} ({job['progress']}%)",
        )

    if format not in ("txt", "json"):
        raise HTTPException(status_code=400, detail="format must be 'txt' or 'json'")

    output_path = get_output_path(job_id, format)
    if not output_path:
        raise HTTPException(
            status_code=404,
            detail=f"Output {format} not found or expired (max 1 hour)",
        )

    media_type = "text/plain; charset=utf-8" if format == "txt" else "application/json"
    filename = f"translation_{job_id}.{format}"

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ============================================================
# DELETE /api/video/delete/{job_id}
# ============================================================
@router.delete("/delete/{job_id}")
async def video_delete(job_id: str):
    """Hapus job video + semua file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    delete_job(job_id)
    print(f"[Video API] Job {job_id} deleted")

    return {"status": "deleted", "job_id": job_id}


# ============================================================
# DELETE /api/video/delete-all
# ============================================================
@router.delete("/delete-all")
async def video_delete_all():
    """Hapus semua job video (cleanup manual)."""
    jobs = list_jobs()
    count = len(jobs)

    for job in jobs:
        delete_job(job["job_id"])

    print(f"[Video API] Deleted all {count} jobs")

    return {"status": "deleted_all", "count": count}
"""
Smart Room Backend
FastAPI + WebSocket + Whisper + NLLB
"""

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import json
import asyncio
import time
from datetime import datetime

from pipeline import process_audio, LANG_MAP
from rooms import manager, Room
from video_api import router as video_router
from video_processor import (
    create_job,
    get_job,
    update_job,
    list_jobs,
    delete_job,
    check_expired_jobs,
    get_output_path,
    process_video,
    video_jobs,
    UPLOAD_DIR,
    MAX_FILE_SIZE_MB,
)
import shutil


app = FastAPI(title="Smart Room Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = "/app/frontend"


# ============================================================
# HELPER: SERVE HTML
# ============================================================
def serve_html(filename: str) -> str:
    path = os.path.join(FRONTEND_DIR, filename)
    if not os.path.exists(path):
        return f"<h1>File {filename} not found</h1>"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# HTTP PAGES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """Homepage - Create Room, Join Room, or Video Translate."""
    return serve_html("index.html")


@app.get("/video", response_class=HTMLResponse)
async def video_page():
    """Video translate testing page."""
    return serve_html("video.html")


@app.get("/host", response_class=HTMLResponse)
async def host_page():
    """Host page."""
    return serve_html("host.html")


@app.get("/join", response_class=HTMLResponse)
async def join_page():
    """Audience page."""
    return serve_html("join.html")


# ============================================================
# HTTP API
# ============================================================
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/rooms")
async def create_room():
    """Create new room, return code."""
    room = manager.create_room()
    return {"code": room.code, "created_at": room.created_at.isoformat()}


@app.get("/api/rooms/{code}")
async def get_room_info(code: str):
    """Get room info."""
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room.get_info()


@app.get("/api/rooms/{code}/history")
async def get_room_history(code: str, lang: str = "id"):
    """Get room history for specific language."""
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"code": code, "lang": lang, "history": room.get_history_for_audience(lang)}


@app.get("/api/rooms/{code}/export")
async def export_room_history(code: str, format: str = "txt"):
    """Download room history."""
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    history = room.get_full_history()

    if format == "json":
        content = json.dumps({
            "room_code": room.code,
            "created_at": room.created_at.isoformat(),
            "host_name": room.host_name,
            "host_lang": room.host_lang,
            "total_entries": len(history),
            "entries": history,
        }, ensure_ascii=False, indent=2)
        return PlainTextResponse(
            content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=room_{code}.json"},
        )

    # Default: txt
    lines = [
        f"Smart Room - {room.code}",
        f"Created: {room.created_at.isoformat()}",
        f"Host: {room.host_name} ({room.host_lang})",
        f"Total: {len(history)} entries",
        "=" * 60,
        "",
    ]
    for i, entry in enumerate(history, 1):
        ts = entry["timestamp"]
        lines.append(f"[{i}] {ts} - {entry.get('speaker', 'host')}")
        lines.append(f"  Original ({entry.get('source_lang', '?')}): {entry['original_text']}")
        for lang_code, text in entry.get("translations", {}).items():
            lang_name = LANG_MAP.get(lang_code, {}).get("name", lang_code)
            lines.append(f"  {lang_name}: {text}")
        lines.append("")

    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=room_{code}.txt"},
    )


@app.delete("/api/rooms/{code}")
async def delete_room(code: str):
    """Close/delete room (host only)."""
    room = manager.get_room(code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")

    # Broadcast room_closed ke semua user
    await room.broadcast({"type": "room_closed", "reason": "Host closed the room"})

    # Hapus room dari memori
    manager.delete_room(code)
    return {"status": "closed", "code": code}


# ============================================================
# WEBSOCKET: HOST
# ============================================================
@app.websocket("/ws/host/{code}")
async def ws_host(websocket: WebSocket, code: str):
    """
    WebSocket for Host.
    Query params: lang (id/en/zh/...), name (host name)
    """
    room = manager.get_room(code)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    # Parse query params
    host_lang = websocket.query_params.get("lang", "ru")
    host_name = websocket.query_params.get("name", "Host")

    if host_lang not in LANG_MAP:
        host_lang = "ru"

    # Set room host info
    await websocket.accept()
    room.host_ws = websocket
    room.host_lang = host_lang
    room.host_name = host_name
    print(f"[Host] Connected to room {code} (name={host_name}, lang={host_lang})")

    try:
        # Kirim info awal
        await websocket.send_json({
            "type": "connected",
            "room_code": code,
            "host_name": host_name,
            "host_lang": host_lang,
            "audiences": len(room.audiences),
        })

        while True:
            data = await websocket.receive()

            # Binary = audio chunk
            if "bytes" in data and data["bytes"]:
                audio_bytes = data["bytes"]
                print(f"[Host] Received {len(audio_bytes)} bytes audio")

                # Proses di background
                asyncio.create_task(
                    process_and_broadcast(
                        room=room,
                        audio_bytes=audio_bytes,
                        speaker_name=room.host_name,
                        source_lang=room.host_lang,
                        speaker_ws=None,  # None = host
                    )
                )

            # Text = control message
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    await handle_host_message(room, msg, websocket)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[Host] Error handling message: {e}")

    except WebSocketDisconnect:
        print(f"[Host] Disconnected from room {code}")
    except RuntimeError as e:
        print(f"[Host] RuntimeError: {e}")
    finally:
        room.host_ws = None


async def handle_host_message(room: Room, msg: dict, websocket: WebSocket):
    """Handle control message dari Host."""
    msg_type = msg.get("type")

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})

    elif msg_type == "host_mic_on":
        # Host mulai bicara
        print(f"[Host] Mic ON")
        await room.broadcast({
            "type": "host_speaking",
            "is_speaking": True,
            "host_name": room.host_name,
        }, exclude_ws=websocket)

    elif msg_type == "host_mic_off":
        # Host selesai bicara
        print(f"[Host] Mic OFF")
        await room.broadcast({
            "type": "host_speaking",
            "is_speaking": False,
            "host_name": room.host_name,
        }, exclude_ws=websocket)

    elif msg_type == "approve_speaker":
        # Host approve audience untuk bicara
        target_name = msg.get("target_name")
        async with room.lock:
            target_ws = None
            for ws, info in room.audiences.items():
                if info.get("name") == target_name:
                    target_ws = ws
                    break

        if not target_ws:
            await websocket.send_json({
                "type": "error",
                "message": f"Audience '{target_name}' not found",
            })
            return

        # Cek apakah bisa approve (tidak ada speaker aktif)
        if room.active_speaker is not None:
            await websocket.send_json({
                "type": "error",
                "message": "Another speaker is already active. Stop them first.",
            })
            return

        # Approve
        success = await room.approve_speaker(target_ws)
        if success:
            print(f"[Host] Approved speaker: {target_name}")

            # Notif ke audience yang di-approve
            try:
                await target_ws.send_json({
                    "type": "speaker_approved",
                    "by": room.host_name,
                })
            except Exception as e:
                print(f"[Host] Error notifying approved speaker: {e}")

            # Broadcast ke semua (termasuk host) kalau ada speaker baru
            await room.broadcast({
                "type": "speaker_approved_broadcast",
                "speaker_name": target_name,
            })

    elif msg_type == "stop_speaker":
        # Host stop audience yang sedang bicara
        async with room.lock:
            speaker_ws = room.active_speaker
            speaker_name = None
            if speaker_ws and speaker_ws in room.audiences:
                speaker_name = room.audiences[speaker_ws].get("name", "Unknown")

        await room.stop_speaker()
        print(f"[Host] Stopped speaker: {speaker_name}")

        # Notif ke audience yang di-stop
        if speaker_ws:
            try:
                await speaker_ws.send_json({
                    "type": "speaker_stopped",
                    "by": room.host_name,
                })
            except Exception:
                pass

        # Broadcast ke semua
        await room.broadcast({
            "type": "speaker_stopped_broadcast",
            "speaker_name": speaker_name or "Unknown",
        })

    elif msg_type == "close_room":
        # Host close room
        print(f"[Host] Closing room {room.code}")
        await room.broadcast({
            "type": "room_closed",
            "reason": "Host closed the room",
        })
        manager.delete_room(room.code)


# ============================================================
# WEBSOCKET: AUDIENCE
# ============================================================
@app.websocket("/ws/audience/{code}")
async def ws_audience(websocket: WebSocket, code: str):
    """
    WebSocket for Audience.
    Query params: lang (id/en/zh/...), name (audience name)
    """
    room = manager.get_room(code)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    # Parse query params
    lang = websocket.query_params.get("lang", "id")
    name = websocket.query_params.get("name", "Anonymous")

    if lang not in LANG_MAP:
        lang = "id"
    if not name or not name.strip():
        name = "Anonymous"

    await websocket.accept()

    # Simpan audience
    async with room.lock:
        room.audiences[websocket] = {
            "lang": lang,
            "name": name,
            "joined_at": datetime.now(),
        }

    print(f"[Audience] '{name}' joined room {code} (lang={lang})")

    try:
        # Kirim info awal + history
        history = room.get_history_for_audience(lang)
        await websocket.send_json({
            "type": "connected",
            "room_code": code,
            "lang": lang,
            "name": name,
            "history": history,
            "host_name": room.host_name,
            "host_lang": room.host_lang,
            "total_audiences": len(room.audiences),
        })

        # Notif ke host
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_joined",
                    "name": name,
                    "lang": lang,
                    "total_audiences": len(room.audiences),
                })
            except Exception:
                pass

        # Main loop
        while True:
            data = await websocket.receive()

            # Binary = audio (hanya kalau audience ini active speaker)
            if "bytes" in data and data["bytes"]:
                if room.active_speaker == websocket:
                    audio_bytes = data["bytes"]
                    print(f"[Audience/{name}] Received {len(audio_bytes)} bytes audio")

                    # Dapatkan bahasa audience
                    async with room.lock:
                        info = room.audiences.get(websocket)
                        source_lang = info["lang"] if info else "id"

                    asyncio.create_task(
                        process_and_broadcast(
                            room=room,
                            audio_bytes=audio_bytes,
                            speaker_name=name,
                            source_lang=source_lang,
                            speaker_ws=websocket,
                        )
                    )

            # Text = control message
            elif "text" in data and data["text"]:
                try:
                    msg = json.loads(data["text"])
                    await handle_audience_message(room, msg, websocket, name, lang)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[Audience/{name}] Error handling message: {e}")

    except WebSocketDisconnect:
        print(f"[Audience] '{name}' left room {code}")
    except RuntimeError as e:
        print(f"[Audience] RuntimeError: {e}")
    finally:
        # Cleanup
        async with room.lock:
            room.audiences.pop(websocket, None)
            room.raised_hands.pop(websocket, None)

            # Kalau audience ini active speaker, stop
            if room.active_speaker == websocket:
                room.active_speaker = None

        # Notif ke host
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_left",
                    "name": name,
                    "total_audiences": len(room.audiences),
                })
            except Exception:
                pass


async def handle_audience_message(
    room: Room,
    msg: dict,
    websocket: WebSocket,
    name: str,
    lang: str,
):
    """Handle control message dari Audience."""
    msg_type = msg.get("type")

    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})

    elif msg_type == "raise_hand":
        # Cek apakah audience ini sudah raise hand
        if websocket in room.raised_hands:
            return  # sudah raise hand, skip

        # Cek apakah audience ini sedang bicara
        if room.active_speaker == websocket:
            return  # sudah bicara, tidak perlu raise hand

        # Tambah ke raised hands
        async with room.lock:
            room.raised_hands[websocket] = {
                "lang": lang,
                "name": name,
                "raised_at": datetime.now(),
            }

        print(f"[Audience/{name}] Raised hand")

        # Notif ke host
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_raised_hand",
                    "name": name,
                    "lang": lang,
                    "total_raised_hands": len(room.raised_hands),
                })
            except Exception:
                pass

        # Konfirmasi ke audience
        await websocket.send_json({
            "type": "raise_hand_confirmed",
        })

    elif msg_type == "lower_hand":
        # Hapus dari raised hands
        async with room.lock:
            room.raised_hands.pop(websocket, None)

        print(f"[Audience/{name}] Lowered hand")

        # Notif ke host
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "audience_lowered_hand",
                    "name": name,
                    "total_raised_hands": len(room.raised_hands),
                })
            except Exception:
                pass

        await websocket.send_json({"type": "lower_hand_confirmed"})

    elif msg_type == "audience_mic_off":
        # Audience selesai bicara
        async with room.lock:
            if room.active_speaker == websocket:
                room.active_speaker = None

        print(f"[Audience/{name}] Mic OFF (self)")

        # Notif ke host & semua
        await room.broadcast({
            "type": "speaker_stopped_broadcast",
            "speaker_name": name,
        })


# ============================================================
# PROCESS AUDIO & BROADCAST
# ============================================================
async def process_and_broadcast(
    room: Room,
    audio_bytes: bytes,
    speaker_name: str,
    source_lang: str,
    speaker_ws,
):
    """
    Process audio (Whisper + NLLB) dan broadcast hasilnya.
    """
    try:
        # Dapatkan target languages dari room
        target_langs = room.get_target_langs()

        if not target_langs:
            print(f"[Process] No target langs, skip")
            return

        # Run pipeline in executor (blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_audio,
            audio_bytes,
            source_lang,
            target_langs,
        )

        if not result["original"]:
            return  # tidak ada teks (atau halusinasi)

        print(f"[Process/{speaker_name}] {result['original'][:80]}...")

        # Simpan ke history
        entry = await room.add_entry(
            speaker=speaker_name,
            original=result["original"],
            source_lang=source_lang,
            translations=result["translations"],
        )

        # Broadcast ke host (dengan bahasa host)
        if room.host_ws:
            try:
                await room.host_ws.send_json({
                    "type": "translation",
                    "timestamp": entry["timestamp"],
                    "speaker": speaker_name,
                    "original_text": entry["original_text"],
                    "source_lang": source_lang,
                    "translated_text": entry["translations"].get(room.host_lang, ""),
                    "lang": room.host_lang,
                })
            except Exception as e:
                print(f"[Broadcast] Error to host: {e}")

        # Broadcast ke audiences (dengan bahasa masing-masing)
        async with room.lock:
            audiences_snapshot = list(room.audiences.items())

        results = await asyncio.gather(
            *[
                ws.send_json({
                    "type": "translation",
                    "timestamp": entry["timestamp"],
                    "speaker": speaker_name,
                    "original_text": entry["original_text"],
                    "source_lang": source_lang,
                    "translated_text": entry["translations"].get(info["lang"], ""),
                    "lang": info["lang"],
                })
                for ws, info in audiences_snapshot
            ],
            return_exceptions=True,
        )

        # Cleanup dead audiences
        async with room.lock:
            for (ws, _), res in zip(audiences_snapshot, results):
                if isinstance(res, Exception):
                    room.audiences.pop(ws, None)
                    room.raised_hands.pop(ws, None)

    except Exception as e:
        print(f"[Process/{speaker_name}] Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# VIDEO TRANSLATE API
# ============================================================

@app.post("/api/video/upload")
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

    # Validasi ukuran file
    # Baca chunk pertama untuk cek ukuran (UploadFile tidak punya .size langsung)
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

    # Simpan file video dengan ekstensi yang sesuai
    ext = "mp4"
    if video.filename and "." in video.filename:
        ext = video.filename.rsplit(".", 1)[-1].lower()
        if ext not in ["mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "3gp"]:
            ext = "mp4"

    video_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")

    # Simpan file (streaming, biar tidak boros RAM)
    try:
        with open(video_path, "wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        # Cleanup kalau gagal
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

    print(f"[Video] Job {job_id} started: {video.filename} ({file_size_mb:.1f} MB)")

    return {
        "job_id": job_id,
        "status": "pending",
        "filename": video.filename,
        "file_size_mb": round(file_size_mb, 1),
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


@app.get("/api/video/status/{job_id}")
async def video_status(job_id: str):
    """Cek status job video."""
    check_expired_jobs()

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # Return tanpa field internal
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


@app.get("/api/video/jobs")
async def video_list_jobs():
    """List semua job video (untuk UI)."""
    check_expired_jobs()

    jobs = list_jobs()
    # Hilangkan field internal
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


@app.get("/api/video/download/{job_id}")
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
        raise HTTPException(status_code=404, detail=f"Output {format} not found or expired")

    # Return file
    media_type = "text/plain; charset=utf-8" if format == "txt" else "application/json"
    filename = f"translation_{job_id}.{format}"

    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()

    return PlainTextResponse(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/api/video/delete/{job_id}")
async def video_delete(job_id: str):
    """Hapus job video + semua file."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    delete_job(job_id)
    print(f"[Video] Job {job_id} deleted")

    return {"status": "deleted", "job_id": job_id}


@app.delete("/api/video/delete-all")
async def video_delete_all():
    """Hapus semua job video (admin only, untuk cleanup manual)."""
    jobs = list_jobs()
    count = len(jobs)

    for job in jobs:
        delete_job(job["job_id"])

    print(f"[Video] Deleted all {count} jobs")

    return {"status": "deleted_all", "count": count}


# ============================================================
# VIDEO TRANSCRIBE (untuk testing)
# ============================================================
@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    source_lang: str = Form("ru"),
    target_lang: str = Form("id"),
):
    """Video/audio translate endpoint (single target)."""
    content = await audio.read()

    # Target langs: source + target
    src_nllb = LANG_MAP.get(source_lang, LANG_MAP["ru"])["nllb"]
    tgt_nllb = LANG_MAP.get(target_lang, LANG_MAP["id"])["nllb"]

    target_langs = {
        source_lang: src_nllb,
        target_lang: tgt_nllb,
    }

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        process_audio,
        content,
        source_lang,
        target_langs,
    )

    return result
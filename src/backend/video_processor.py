"""
Video Processor
Handle upload, extract audio, chunk, transcribe, translate.
Support video up to 3 hours.

Strategy:
- Video/audio/chunk: deleted after processing complete
- Output (TXT + JSON): kept for 1 hour
- Auto-cleanup: on-demand (checked on each request)
"""

import os
import uuid
import json
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline import process_audio, LANG_MAP


# ============================================================
# CONSTANTS
# ============================================================
UPLOAD_DIR = "/app/output/uploads"
AUDIO_DIR = "/app/audio"
OUTPUT_DIR = "/app/output/translations"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_DURATION_SEC = 300          # 5 menit
MAX_VIDEO_DURATION_SEC = 3 * 3600 # 3 jam
MAX_FILE_SIZE_MB = 1024           # 1 GB
OUTPUT_TTL_SECONDS = 3600         # 1 jam


# ============================================================
# GLOBAL JOB STORAGE (in-memory)
# ============================================================
video_jobs = {}  # {job_id: {status, progress, message, ...}}


# ============================================================
# HELPERS: FFMPEG
# ============================================================
def run_ffmpeg(args: list, timeout: int = 3600) -> bool:
    """Run ffmpeg command. Return True kalau sukses."""
    try:
        subprocess.run(
            ["ffmpeg"] + args,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FFmpeg] Error: {e.stderr.decode()[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"[FFmpeg] Timeout after {timeout}s")
        return False
    except FileNotFoundError:
        print(f"[FFmpeg] Not found in PATH")
        return False


def get_audio_duration(audio_path: str) -> float:
    """Dapatkan durasi audio (detik) pakai ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[FFprobe] Error: {e}")
        return 0.0


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Ekstrak audio dari video, convert ke WAV 16kHz mono."""
    return run_ffmpeg([
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        audio_path,
    ], timeout=3600)


def split_audio(audio_path: str, output_dir: str, job_id: str, chunk_sec: int = 300) -> list:
    """Potong audio jadi chunk. Return list of chunk paths."""
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return []

    chunks = []
    num_chunks = int(duration // chunk_sec) + (1 if duration % chunk_sec > 0 else 0)

    for i in range(num_chunks):
        start = i * chunk_sec
        chunk_path = os.path.join(output_dir, f"{job_id}_chunk_{i:03d}.wav")

        success = run_ffmpeg([
            "-i", audio_path,
            "-ss", str(start),
            "-t", str(chunk_sec),
            "-acodec", "copy",
            "-y",
            chunk_path,
        ], timeout=600)

        if success and os.path.exists(chunk_path):
            chunks.append(chunk_path)

    return chunks


# ============================================================
# CLEANUP
# ============================================================
def cleanup_processing_files(job_id: str):
    """
    Hapus file video/audio/chunk.
    Dipanggil setelah processing selesai.
    """
    # Chunks
    for f in Path(AUDIO_DIR).glob(f"{job_id}_chunk_*.wav"):
        try:
            f.unlink()
        except Exception:
            pass

    # Audio utama
    audio_path = os.path.join(AUDIO_DIR, f"{job_id}.wav")
    if os.path.exists(audio_path):
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    # Video (coba semua ekstensi)
    for ext in ["mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v", "3gp"]:
        video_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except Exception:
                pass


def delete_job(job_id: str):
    """Hapus job + semua file (termasuk output)."""
    if job_id in video_jobs:
        del video_jobs[job_id]
    cleanup_processing_files(job_id)
    for ext in ["txt", "json"]:
        p = os.path.join(OUTPUT_DIR, f"{job_id}.{ext}")
        if os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass


def check_expired_jobs():
    """
    Cek job yang sudah selesai > 1 jam, hapus.
    On-demand: dipanggil saat ada request /api/video/*.
    """
    now = datetime.now()
    expired = []

    for job_id, job in list(video_jobs.items()):
        completed = job.get("completed_at")
        if not completed:
            continue
        try:
            age_sec = (now - datetime.fromisoformat(completed)).total_seconds()
            if age_sec > OUTPUT_TTL_SECONDS:
                expired.append(job_id)
        except Exception:
            pass

    for job_id in expired:
        print(f"[Cleanup] Removing expired job {job_id}")
        delete_job(job_id)

    return len(expired)


# ============================================================
# JOB MANAGEMENT
# ============================================================
def create_job(filename: str, source_lang: str, target_lang: str) -> str:
    """Create new video job, return job_id."""
    job_id = uuid.uuid4().hex[:12]
    video_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Job created",
        "filename": filename,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "completed_at": None,
        "duration": 0,
        "total_chunks": 0,
        "current_chunk": 0,
    }
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Get job status."""
    return video_jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id not in video_jobs:
        return
    video_jobs[job_id].update(kwargs)
    video_jobs[job_id]["updated_at"] = datetime.now().isoformat()


def list_jobs() -> list:
    """List all jobs (sorted by created_at desc)."""
    jobs = list(video_jobs.values())
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


def get_output_path(job_id: str, fmt: str) -> Optional[str]:
    """Return path ke file output (txt atau json)."""
    if fmt not in ("txt", "json"):
        return None
    p = os.path.join(OUTPUT_DIR, f"{job_id}.{fmt}")
    return p if os.path.exists(p) else None


# ============================================================
# MAIN PROCESSING
# ============================================================
async def process_video(
    job_id: str,
    video_path: str,
    source_lang: str,
    target_lang: str,
):
    """Main video processing pipeline."""
    try:
        # -------- Step 1: Extract audio --------
        update_job(job_id, status="extracting", progress=5, message="Extracting audio...")

        audio_path = os.path.join(AUDIO_DIR, f"{job_id}.wav")
        success = await asyncio.get_event_loop().run_in_executor(
            None, extract_audio, video_path, audio_path
        )

        if not success:
            update_job(job_id, status="error", message="Failed to extract audio")
            return

        duration = await asyncio.get_event_loop().run_in_executor(
            None, get_audio_duration, audio_path
        )

        if duration > MAX_VIDEO_DURATION_SEC:
            update_job(
                job_id,
                status="error",
                message=f"Video too long: {duration/3600:.1f}h (max 3h)",
            )
            return

        update_job(
            job_id,
            progress=10,
            message=f"Audio extracted ({duration/60:.1f} min). Splitting...",
            duration=duration,
        )

        # -------- Step 2: Split audio --------
        chunks = await asyncio.get_event_loop().run_in_executor(
            None, split_audio, audio_path, AUDIO_DIR, job_id, CHUNK_DURATION_SEC
        )

        if not chunks:
            update_job(job_id, status="error", message="Failed to split audio")
            return

        num_chunks = len(chunks)
        update_job(
            job_id,
            status="transcribing",
            progress=15,
            message=f"Transcribing {num_chunks} chunks...",
            total_chunks=num_chunks,
            current_chunk=0,
        )

        # -------- Step 3: Transcribe each chunk --------
        transcript_parts = []
        target_langs = {target_lang: LANG_MAP[target_lang]["nllb"]}

        for i, chunk_path in enumerate(chunks):
            update_job(
                job_id,
                current_chunk=i + 1,
                message=f"Transcribing chunk {i+1}/{num_chunks}...",
                progress=15 + int((i / num_chunks) * 65),  # 15% -> 80%
            )

            try:
                with open(chunk_path, "rb") as f:
                    chunk_bytes = f.read()

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    process_audio,
                    chunk_bytes,
                    source_lang,
                    target_langs,
                )

                if result["original"]:
                    transcript_parts.append({
                        "chunk_index": i,
                        "start_sec": i * CHUNK_DURATION_SEC,
                        "original_text": result["original"],
                        "translated_text": result["translations"].get(target_lang, ""),
                    })
            except Exception as e:
                print(f"[Video/{job_id}] Chunk {i} error: {e}")
                continue

        if not transcript_parts:
            update_job(job_id, status="error", message="No transcription result")
            return

        # -------- Step 4: Finalize --------
        update_job(
            job_id,
            status="finalizing",
            progress=85,
            message="Finalizing output...",
        )

        full_original = " ".join(p["original_text"] for p in transcript_parts)
        full_translated = " ".join(p["translated_text"] for p in transcript_parts)

        # -------- Step 5: Save output (TXT + JSON) --------
        metadata = {
            "job_id": job_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "duration_sec": round(duration, 1),
            "duration_min": round(duration / 60, 1),
            "total_chunks": num_chunks,
            "created_at": video_jobs[job_id]["created_at"],
            "completed_at": datetime.now().isoformat(),
        }

        # JSON output
        json_data = {
            **metadata,
            "original_text": full_original,
            "translated_text": full_translated,
            "chunks": transcript_parts,
        }
        json_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # TXT output
        txt_lines = [
            f"Smart Room - Video Translation",
            f"================================",
            f"Job ID     : {job_id}",
            f"Source Lang: {source_lang}",
            f"Target Lang: {target_lang}",
            f"Duration   : {metadata['duration_min']} minutes",
            f"Chunks     : {num_chunks}",
            f"Created    : {metadata['created_at']}",
            f"Completed  : {metadata['completed_at']}",
            f"",
            f"--- ORIGINAL TEXT ---",
            full_original,
            f"",
            f"--- TRANSLATED TEXT ({target_lang}) ---",
            full_translated,
            f"",
            f"--- SEGMENTS ---",
        ]
        for p in transcript_parts:
            start_min = p["start_sec"] // 60
            start_sec = p["start_sec"] % 60
            txt_lines.append(f"[{start_min:02d}:{start_sec:02d}]")
            txt_lines.append(f"  Original  : {p['original_text']}")
            txt_lines.append(f"  Translated: {p['translated_text']}")
            txt_lines.append("")

        txt_path = os.path.join(OUTPUT_DIR, f"{job_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        # Update job
        update_job(
            job_id,
            status="done",
            progress=100,
            message="Translation complete!",
            completed_at=datetime.now().isoformat(),
        )

        # -------- Step 6: CLEANUP processing files --------
        await asyncio.get_event_loop().run_in_executor(
            None, cleanup_processing_files, job_id
        )

        print(f"[Video/{job_id}] Done! Duration: {duration/60:.1f} min")

    except Exception as e:
        print(f"[Video/{job_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        update_job(job_id, status="error", message=f"Fatal error: {str(e)}")
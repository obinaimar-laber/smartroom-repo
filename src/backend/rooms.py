"""
Room management for Smart Room.
Handles room state, participants, raised hands, active speaker, and history.
"""

import json
import os
import time
import random
import string
import asyncio
from datetime import datetime
from typing import Optional, Set, Dict, Any


# ============================================================
# CONSTANTS
# ============================================================
HISTORY_DIR = "/app/output/rooms"
os.makedirs(HISTORY_DIR, exist_ok=True)

MAX_HISTORY_IN_MEMORY = 500  # max entries in memory (untuk hemat RAM)


# ============================================================
# UTILITIES
# ============================================================
def generate_room_code(length: int = 4) -> str:
    """Generate unique room code (lowercase + digits)."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


# ============================================================
# ROOM CLASS
# ============================================================
class Room:
    """
    Represents a single room with:
    - Host (1 WebSocket)
    - Audiences (multiple WebSockets)
    - Raised hands queue
    - Active speaker (1 audience at a time)
    - Translation history
    """

    def __init__(self, code: str):
        self.code = code
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Host
        self.host_ws = None
        self.host_lang = "ru"       # default, akan di-override saat connect
        self.host_name = "Host"

        # Audiences: {websocket: {"lang": str, "name": str, "joined_at": datetime}}
        self.audiences: Dict[Any, Dict[str, Any]] = {}

        # Raised hands: {websocket: {"lang": str, "name": str, "raised_at": datetime}}
        self.raised_hands: Dict[Any, Dict[str, Any]] = {}

        # Active speaker: WebSocket audience yang sedang bicara (None kalau tidak ada)
        self.active_speaker = None

        # History: list of translation entries
        self.history = []

        # Concurrency lock untuk modifikasi state
        self.lock = asyncio.Lock()

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------
    async def add_entry(
        self,
        speaker: str,
        original: str,
        source_lang: str,
        translations: dict,
    ) -> dict:
        """Tambah entry baru ke history."""
        async with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "speaker": speaker,
                "original_text": original,
                "source_lang": source_lang,
                "translations": translations,
            }
            self.history.append(entry)
            self.last_activity = datetime.now()

            # Batasi history di memori
            if len(self.history) > MAX_HISTORY_IN_MEMORY:
                self.history = self.history[-MAX_HISTORY_IN_MEMORY:]

            return entry

    def get_history_for_audience(self, lang: str) -> list:
        """Ambil history dalam format yang dibutuhkan audiens."""
        result = []
        for entry in self.history:
            result.append({
                "timestamp": entry["timestamp"],
                "speaker": entry.get("speaker", "host"),
                "original_text": entry["original_text"],
                "source_lang": entry.get("source_lang", "ru"),
                "translated_text": entry.get("translations", {}).get(lang, ""),
            })
        return result

    def get_full_history(self) -> list:
        """Return full history (untuk download)."""
        return self.history

    # --------------------------------------------------------
    # TARGET LANGS
    # --------------------------------------------------------
    def get_target_langs(self) -> dict:
        """
        Kumpulkan semua bahasa unik dari Host + Audience.
        Return: dict {lang_code: nllb_code}
        """
        from pipeline import LANG_MAP

        all_langs = {self.host_lang}
        for info in self.audiences.values():
            all_langs.add(info["lang"])

        result = {}
        for lang in all_langs:
            if lang in LANG_MAP:
                result[lang] = LANG_MAP[lang]["nllb"]
        return result

    # --------------------------------------------------------
    # BROADCAST
    # --------------------------------------------------------
    async def broadcast(self, message: dict, exclude_ws=None):
        """
        Kirim message ke Host + semua Audience (kecuali exclude_ws).
        Handle error dan cleanup audiens yang mati.
        """
        # Kirim ke Host
        if self.host_ws and self.host_ws != exclude_ws:
            try:
                await self.host_ws.send_json(message)
            except Exception as e:
                print(f"[Room {self.code}] Error to host: {e}")
                self.host_ws = None

        # Kirim ke Audience (paralel)
        async with self.lock:
            audiences_snapshot = list(self.audiences.items())

        if not audiences_snapshot:
            return

        results = await asyncio.gather(
            *[
                ws.send_json(message)
                for ws, _ in audiences_snapshot
                if ws != exclude_ws
            ],
            return_exceptions=True,
        )

        # Cleanup audiens yang error
        async with self.lock:
            for (ws, _), result in zip(
                [(w, i) for w, i in audiences_snapshot if w != exclude_ws],
                results,
            ):
                if isinstance(result, Exception):
                    self.audiences.pop(ws, None)
                    self.raised_hands.pop(ws, None)

    # --------------------------------------------------------
    # SPEAKER MANAGEMENT
    # --------------------------------------------------------
    async def approve_speaker(self, ws) -> bool:
        """
        Approve audience untuk bicara.
        Return True kalau berhasil, False kalau sudah ada speaker aktif.
        """
        async with self.lock:
            if self.active_speaker is not None:
                return False  # sudah ada yang bicara
            if ws not in self.audiences:
                return False  # bukan audience
            self.active_speaker = ws
            self.raised_hands.pop(ws, None)  # hapus dari daftar raise hand
            return True

    async def stop_speaker(self):
        """Stop audience yang sedang bicara."""
        async with self.lock:
            self.active_speaker = None

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------
    def get_info(self) -> dict:
        """Info ringkas tentang room."""
        return {
            "code": self.code,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "host_connected": self.host_ws is not None,
            "host_name": self.host_name,
            "host_lang": self.host_lang,
            "total_audiences": len(self.audiences),
            "total_raised_hands": len(self.raised_hands),
            "active_speaker": self.active_speaker is not None,
            "total_entries": len(self.history),
            "audiences": [
                {"lang": info["lang"], "name": info.get("name", "Anonymous")}
                for info in self.audiences.values()
            ],
            "raised_hands_list": [
                {
                    "name": info.get("name", "Anonymous"),
                    "lang": info["lang"],
                    "raised_at": info.get("raised_at", "").isoformat()
                    if isinstance(info.get("raised_at"), datetime) else "",
                }
                for info in self.raised_hands.values()
            ],
        }


# ============================================================
# ROOM MANAGER
# ============================================================
class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}

    def create_room(self) -> Room:
        """Create new room with unique code."""
        for _ in range(10):
            code = generate_room_code()
            if code not in self.rooms:
                break
        else:
            raise Exception("Failed to generate unique room code")

        room = Room(code)
        self.rooms[code] = room
        print(f"[RoomManager] Room {code} created")
        return room

    def get_room(self, code: str) -> Optional[Room]:
        return self.rooms.get(code)

    def delete_room(self, code: str):
        """Hapus room dari memori (history hilang)."""
        room = self.rooms.get(code)
        if room:
            del self.rooms[code]
            print(f"[RoomManager] Room {code} deleted")


# Singleton
manager = RoomManager()
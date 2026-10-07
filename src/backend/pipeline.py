"""
Pipeline AI: Audio -> Whisper (speech-to-text) -> NLLB (translation)
Support 8 bahasa: ru, en, zh, es, fa, id, vi, bn
"""

import re
import tempfile
import os
import requests
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import ctranslate2

# ============================================================
# LOAD MODELS
# ============================================================
print("[Pipeline] Loading NLLB model...")
NLLB_PATH = snapshot_download("olob0/nllb-200-distilled-600M-ct2-int8_float16")
tokenizer = AutoTokenizer.from_pretrained(NLLB_PATH)
translator = ctranslate2.Translator(
    NLLB_PATH,
    device="cuda",
    compute_type="int8_float16"
)
print("[Pipeline] NLLB loaded.")


# ============================================================
# LANGUAGE MAPPING (8 bahasa)
# ============================================================
# whisper: kode bahasa untuk Whisper
# nllb: kode bahasa untuk NLLB (FLORES-200)
# name: nama tampilan
# ============================================================
LANG_MAP = {
    "ru": {"whisper": "ru", "nllb": "rus_Cyrl", "name": "Rusia"},
    "en": {"whisper": "en", "nllb": "eng_Latn", "name": "Inggris"},
    "zh": {"whisper": "zh", "nllb": "zho_Hans", "name": "Tionghoa"},
    "es": {"whisper": "es", "nllb": "spa_Latn", "name": "Spanyol"},
    "fa": {"whisper": "fa", "nllb": "pes_Arab", "name": "Persia"},
    "id": {"whisper": "id", "nllb": "ind_Latn", "name": "Indonesia"},
    "vi": {"whisper": "vi", "nllb": "vie_Latn", "name": "Vietnam"},
    "bn": {"whisper": "bn", "nllb": "ben_Beng", "name": "Bangladesh"},
}

# NLLB code untuk bahasa (dipakai untuk lookup)
NLLB_TO_LANG = {v["nllb"]: k for k, v in LANG_MAP.items()}

# URL Whisper service (di dalam Docker network)
WHISPER_URL = os.getenv("WHISPER_URL", "http://whisper:9000/asr")


# ============================================================
# HALLUCINATION FILTER
# ============================================================
# Whisper sering menghasilkan teks sampah saat ada noise/silence.
# Filter ini menghapus teks yang tidak masuk akal.
# ============================================================
HALLUCINATION_PATTERNS = [
    # Rusia
    "тьфу", "субтитры", "редактор", "продолжение следует",
    "ставим лайки", "подписывайтесь на канал", "субтитры сделал",
    "продолжение в следующей части", "dimatorzok",
    # Inggris
    "thank you for watching", "subscribe", "amara.org",
    "thanks for watching", "please subscribe", "like and subscribe",
    # Umum
    "[музыка]", "[аплодисменты]", "[смех]", "♪", "…",
    "music", "applause", "laughter",
    # Cina
    "感谢观看", "请订阅", "字幕由", "谢谢观看",
]


def is_hallucination(text: str) -> bool:
    """
    Cek apakah teks hasil Whisper adalah halusinasi.
    Return True kalau teks dianggap sampah.
    """
    if not text or len(text.strip()) < 2:
        return True

    text_lower = text.lower()

    # Cek pattern halusinasi umum
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in text_lower:
            return True

    # Terlalu banyak karakter non-alfanumerik (misal: "Ph.a≡♦ит")
    non_alpha = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:\'"-–—()[]')
    if len(text) > 5 and non_alpha / len(text) > 0.3:
        return True

    # Teks yang hanya berisi 1 kata berulang (misal: "тьфу тьфу тьфу")
    words = text.strip().split()
    if len(words) >= 3:
        # Cek apakah 3 kata pertama sama semua
        if len(set(words[:3])) == 1:
            return True
        # Cek apakah semua kata sama
        if len(set(words)) == 1:
            return True

    return False


# ============================================================
# TRANSLATE (NLLB)
# ============================================================
def translate(text: str, src_nllb: str, tgt_nllb: str) -> str:
    """
    Terjemahkan teks dari src_nllb ke tgt_nllb.
    Split per kalimat untuk akurasi lebih baik.
    """
    if not text.strip():
        return ""

    # Kalau bahasa sumber = target, tidak perlu translate
    if src_nllb == tgt_nllb:
        return text

    # Split berdasarkan tanda baca akhir kalimat
    sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return ""

    results = []
    tokenizer.src_lang = src_nllb

    for sentence in sentences:
        try:
            tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence).input_ids)
            result = translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_nllb]],
                max_batch_size=1,
                beam_size=1,  # greedy, lebih cepat
            )
            output_tokens = result[0].hypotheses[0][1:]  # skip target prefix
            translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))
            results.append(translated)
        except Exception as e:
            print(f"[Translate] Error on sentence '{sentence[:50]}...': {e}")
            results.append("")  # skip kalimat yang error

    return " ".join(r for r in results if r)


def translate_all(text: str, src_nllb: str, target_langs: dict) -> dict:
    """
    Terjemahkan ke semua bahasa target.

    Args:
        text: teks asli
        src_nllb: kode NLLB bahasa sumber (misal "ind_Latn")
        target_langs: dict {lang_code: nllb_code}, misal {"en": "eng_Latn", "zh": "zho_Hans"}

    Returns:
        dict {lang_code: translated_text}
    """
    translations = {}

    for lang_code, nllb_code in target_langs.items():
        if nllb_code == src_nllb:
            # Bahasa sumber: tidak perlu diterjemahkan
            translations[lang_code] = text
        else:
            translations[lang_code] = translate(text, src_nllb, nllb_code)

    return translations


# ============================================================
# WHISPER (SPEECH-TO-TEXT)
# ============================================================
def transcribe_audio(audio_bytes: bytes, source_lang: str = "ru") -> str:
    """
    Kirim audio ke Whisper, dapat teks asli.

    Args:
        audio_bytes: raw audio file content (webm/mp3/wav)
        source_lang: kode bahasa (ru, en, zh, dll). Kalau None, Whisper auto-detect.

    Returns:
        Teks hasil transkripsi
    """
    # Simpan ke file temporary
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        params = {"task": "transcribe", "output": "json"}
        if source_lang and source_lang in LANG_MAP:
            params["language"] = LANG_MAP[source_lang]["whisper"]

        with open(tmp_path, "rb") as f:
            files = {"audio_file": f}
            r = requests.post(WHISPER_URL, params=params, files=files, timeout=120)
        r.raise_for_status()
        result = r.json()
        return result.get("text", "").strip()

    except requests.exceptions.Timeout:
        print(f"[Whisper] Timeout after 120s")
        return ""
    except requests.exceptions.ConnectionError as e:
        print(f"[Whisper] Connection error: {e}")
        return ""
    except Exception as e:
        print(f"[Whisper] Error: {e}")
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ============================================================
# PIPELINE LENGKAP
# ============================================================
def process_audio(audio_bytes: bytes, source_lang: str, target_langs: dict) -> dict:
    """
    Pipeline lengkap: audio -> Whisper -> NLLB -> hasil terjemahan.

    Args:
        audio_bytes: audio dari user
        source_lang: bahasa yang digunakan speaker (id, en, zh, dll)
        target_langs: dict {lang_code: nllb_code} bahasa target di room

    Returns:
        {
            "original": str,
            "source_lang": str,
            "translations": {lang_code: text}
        }
    """
    # Validasi source_lang
    if source_lang not in LANG_MAP:
        print(f"[Pipeline] Unknown source_lang: {source_lang}, default to 'ru'")
        source_lang = "ru"

    # Step 1: Whisper
    original_text = transcribe_audio(audio_bytes, source_lang)

    if not original_text:
        return {"original": "", "source_lang": source_lang, "translations": {}}

    # Step 2: Filter halusinasi
    if is_hallucination(original_text):
        print(f"[Pipeline] Hallucination filtered: '{original_text[:60]}'")
        return {"original": "", "source_lang": source_lang, "translations": {}}

    print(f"[Pipeline] Transcribed: '{original_text[:80]}'")

    # Step 3: Translate ke semua bahasa target
    src_nllb = LANG_MAP[source_lang]["nllb"]
    translations = translate_all(original_text, src_nllb, target_langs)

    return {
        "original": original_text,
        "source_lang": source_lang,
        "translations": translations,
    }


# ============================================================
# UTILITY: Get NLLB code from lang code
# ============================================================
def get_nllb_code(lang: str) -> str:
    """Return NLLB code untuk bahasa. Default: rus_Cyrl."""
    return LANG_MAP.get(lang, LANG_MAP["ru"])["nllb"]


def get_lang_name(lang: str) -> str:
    """Return nama tampilan bahasa."""
    return LANG_MAP.get(lang, {"name": lang})["name"]


# ============================================================
# TEST MANUAL (jalankan file ini langsung untuk test)
# ============================================================
if __name__ == "__main__":
    print("\n=== Test Pipeline ===")
    # Test filter halusinasi
    test_texts = [
        "Привет, как дела?",
        "тьфу тьфу тьфу",
        "thank you for watching",
        "Ph.a≡♦ит",
        "Halo apa kabar",
    ]
    for t in test_texts:
        result = "HALLUCINATION" if is_hallucination(t) else "OK"
        print(f"  [{result}] {t}")
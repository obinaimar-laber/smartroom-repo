Smart Room Documentation
========================

Smart Room is an interactive translator application for multilingual classrooms.
It helps international students understand lectures taught in foreign languages
(especially Russian) by providing real-time translation.

Features
--------

- Real-time translation (Host + Audience)
- Raise hand for speaking permission
- Video translate (up to 3 hours)
- Support 8 languages: ru, en, zh, es, fa, id, vi, bn
- Download history (TXT/JSON)

Tech Stack
----------

- **Backend:** Python 3.11, FastAPI, WebSocket
- **AI Speech-to-Text:** Whisper (OpenAI)
- **AI Translation:** NLLB-200 (Facebook)
- **Frontend:** HTML5, CSS3, JavaScript
- **Containerization:** Docker Compose
- **GPU:** NVIDIA CUDA 12.4

Design Decision: No Inheritance
-------------------------------

Proyek Smart Room menggunakan **composition over inheritance** 
untuk mengelola state user. Setiap user direpresentasikan 
sebagai dictionary di dalam class ``Room``, bukan sebagai 
class terpisah.

**Alasan:**

1. WebSocket lifecycle bersifat dinamis (connect, active, disconnect).
2. State user sederhana (cukup ``name`` dan ``lang``).
3. Pythonic: composition lebih fleksibel dari inheritance.
4. Tidak ada behavior spesifik per user role — semua logic ada di ``Room``.

**Konsekuensi:**

Requirement "derived class" tidak dipenuhi karena design decision ini.
Namun, konsep OOP (encapsulation, composition) tetap diterapkan
di class ``Room`` dan ``RoomManager``.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   autoapi/index
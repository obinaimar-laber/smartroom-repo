# ================================================================
# DOCKER COMPOSE - SMART ROOM
# ================================================================
# File ini mendefinisikan 3 komponen utama:
#   1. SERVICE WHISPER  -> Speech-to-Text (audio -> teks)
#   2. SERVICE BACKEND  -> FastAPI + NLLB (terjemahan + WebSocket)
#   3. NETWORK          -> Jaringan internal antar container
# ================================================================
# Cara pakai:
#   docker compose up -d       (jalankan semua)
#   docker compose down        (hentikan semua)
#   docker compose logs -f     (lihat log)
#   docker compose ps          (lihat status)
# ================================================================


# ================================================================
# SERVICE 1: WHISPER
# ================================================================
# Fungsi : Menerima audio dari backend, mengubahnya jadi teks.
# Image  : onerahmet/openai-whisper-asr-webservice:latest-gpu
# Port   : 9000 (internal), di-expose ke host di port 9000
# GPU    : Ya (butuh CUDA)
# Model  : small (bisa diganti: tiny, base, small, medium, large)
# ================================================================
services:
  whisper:
    image: onerahmet/openai-whisper-asr-webservice:latest-gpu
    container_name: whisper
    ports:
      - "9000:9000"                    # host:container
    volumes:
      - ./audio:/data                  # folder audio di host -> /data di container
    environment:
      - ASR_MODEL=small                # model Whisper
      - ASR_ENGINE=openai_whisper      # engine yang dipakai
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]      # reservasi GPU
    networks:
      - smartroom-net                  # gabung ke network smartroom-net
    restart: unless-stopped            # auto-restart kalau crash
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 30s                    # cek tiap 30 detik
      timeout: 10s                     # timeout 10 detik
      retries: 5                       # coba 5x sebelum dianggap gagal
      start_period: 300s               # beri waktu 5 menit untuk startup awal
                                       # (karena harus download model Whisper)


# ================================================================
# SERVICE 2: BACKEND
# ================================================================
# Fungsi : FastAPI yang menjalankan NLLB (translator) + WebSocket.
# Image  : smartroom-backend:v1
# Port   : 8000 (internal), di-expose ke host di port 8000
# GPU    : Ya (butuh CUDA untuk NLLB)
# Volume : src (kode), models (cache HuggingFace), output (history)
# ================================================================
  backend:
    image: smartroom-backend:v1
    container_name: backend
    ports:
      - "8000:8000"                    # host:container
    volumes:
      - ./src:/app                     # kode backend di host -> /app di container
      - ./models:/root/.cache/huggingface  # cache model NLLB
      - ./output:/app/output           # history room tersimpan di host
    working_dir: /app/backend          # direktori kerja di dalam container
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    environment:
      - WHISPER_URL=http://whisper:9000/asr   # URL internal ke service whisper
      - CUDA_VISIBLE_DEVICES=0         # pakai GPU 0 (RTX 4050)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]      # reservasi GPU
    depends_on:
      whisper:
        condition: service_healthy     # tunggu whisper sehat dulu baru start
    networks:
      - smartroom-net                  # gabung ke network yang sama dengan whisper
    restart: unless-stopped            # auto-restart kalau crash


# ================================================================
# NETWORK
# ================================================================
# Fungsi : Jaringan internal agar backend bisa akses whisper
#          dengan hostname "whisper" (bukan localhost).
# Driver : bridge (default, cukup untuk lokal)
# ================================================================
networks:
  smartroom-net:
    driver: bridge


----------------------------
# 1. Generate ulang (langsung, tanpa hapus)
cd C:\Users\User\Documents\smartroom
mkcert -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1 ::1 10.165.27.171

# 2. Restart backend
docker compose restart backend

# 3. Test di laptop
# Buka: https://localhost:8443/

# 4. Test di HP
# Buka: https://10.165.27.171:8443/
------------------------------------------
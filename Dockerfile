# 1. Temel Sistem: Python 3.10 yüklü hafif Linux
FROM python:3.10-slim

# 2. RDKit ve Grafik İşlemleri için Gerekli Linux Kütüphanelerini Yükle
# (Hugging Face'te aldığın hatayı çözen kısım burasıdır)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Sunucuda 'app' diye bir klasör aç ve içine gir
WORKDIR /app

# 4. Kütüphane listesini kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Senin bilgisayarındaki TÜM dosyaları sunucuya kopyala
COPY . .

# 6. Streamlit'in portunu (kapısını) aç
EXPOSE 8501

# 7. Uygulamayı başlat
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
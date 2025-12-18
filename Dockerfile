# 1. Temel Sistem
FROM python:3.10-slim

# 2. Linux Kütüphanelerini Yükle
# DÜZELTME: 'libgl1-mesa-glx' yerine 'libgl1' yazdık.
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-dev \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma Klasörü
WORKDIR /app

# 4. Kütüphaneleri Yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Dosyaları Kopyala
COPY . .

# 6. Portu Aç
EXPOSE 8501

# 7. Başlat
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]

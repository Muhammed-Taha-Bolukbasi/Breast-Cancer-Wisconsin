# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# Önce requirements dosyasını kopyala ve bağımlılıkları kur
COPY requirements.txt ./

# Sadece gerekli build araçlarını kur ve temizle
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Sadece gerekli dosyaları kopyala
COPY ./src ./src
COPY ./main.py ./
COPY ./conf.yaml ./
COPY ./data ./data

# Stage 2: Çalışma ortamı - sadece runtime gereksinimlerini içerir
FROM python:3.13-slim

WORKDIR /app

# PATH'e ~/.local/bin ekle
ENV PATH="/root/.local/bin:$PATH"

# Sadece runtime bağımlılıklarını kopyala ve kur
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Builder'dan sadece gerekli dosyaları kopyala
COPY --from=builder /app/src ./src
COPY --from=builder /app/main.py ./
COPY --from=builder /app/conf.yaml ./
COPY --from=builder /app/data ./data

# Streamlit uygulamasını başlat
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]

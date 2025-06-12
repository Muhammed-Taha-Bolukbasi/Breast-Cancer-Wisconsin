# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Final image
FROM python:3.13-slim

WORKDIR /app

# PATH’e ~/.local/bin ekle (streamlit burada olabilir)
ENV PATH="/root/.local/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app /app

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
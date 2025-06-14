# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# Copy requirements file and install dependencies first
COPY requirements.txt ./

# Install only necessary build tools and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --upgrade pip && \
    pip install --progress-bar off --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY ./src ./src
COPY ./main.py ./
COPY ./conf.yaml ./
COPY ./data ./data

# Stage 2: Runtime environment - contains only runtime dependencies
FROM python:3.13-slim

WORKDIR /app

# Add ~/.local/bin to PATH (for streamlit etc.)
ENV PATH="/root/.local/bin:$PATH"

# Copy and install only runtime dependencies
COPY requirements.txt ./
RUN pip install --progress-bar off --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy only necessary files from builder
COPY --from=builder /app/src ./src
COPY --from=builder /app/main.py ./
COPY --from=builder /app/conf.yaml ./
COPY --from=builder /app/data ./data

# Start the Streamlit app
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]

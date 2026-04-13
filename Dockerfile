# Dockerfile
# Backend container ONLY — Ollama stays on host

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your existing project files
COPY app.py .
COPY ollama_client.py .
COPY database.py .
COPY profiler.py .
COPY monitor.py .
COPY benchmark.py .
COPY ctx_tuner.py .
COPY config.py .
COPY .env.example .

# Create runtime dirs
RUN mkdir -p /app/data /app/logs

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN printf "[server]\nheadless=true\nport=8501\naddress=0.0.0.0\nenableCORS=false\nenableXsrfProtection=false\n\n[browser]\ngatherUsageStats=false\n" \
> /root/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
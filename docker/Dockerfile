# Amharic ASR service image (CPU). Build from the project root:
#   docker build -f docker/Dockerfile -t eaii-asr:0.1 .
# The model folder is mounted at run time (docker-compose.yml), so one image serves any version.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-serve.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements-serve.txt
COPY src/ src/
COPY configs/ configs/
RUN useradd -m asr && mkdir -p /app/logs && chown asr /app/logs
USER asr
EXPOSE 8000
CMD ["uvicorn", "asr.service:app", "--host", "0.0.0.0", "--port", "8000"]

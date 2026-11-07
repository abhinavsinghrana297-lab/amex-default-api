# ---- 1. Start from a small Linux with Python 3.13 (same Python as training) ----
FROM python:3.13-slim

# LightGBM needs the OpenMP library, which the slim image leaves out
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 2. Install libraries first: this layer is cached and only rebuilds when requirements change ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 3. Put the model inside the image: download the pinned GitHub Release (v1.0.0) ----
COPY scripts/download_model.py scripts/
RUN python scripts/download_model.py

# ---- 4. Copy the application code (changes most often, so it comes last) ----
COPY app/ app/

# ---- 5. Run as a normal user, not root (safer if the service is ever attacked) ----
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 0.0.0.0 = accept connections from outside the container (127.0.0.1 would only accept from inside)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

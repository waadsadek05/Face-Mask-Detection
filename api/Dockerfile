FROM python:3.10-slim

WORKDIR /app

# مكتبات النظام اللازمة لـ OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt /app/requirements.txt

# Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy app + static frontend + model
COPY api/app.py        /app/app.py
COPY api/static        /app/static
COPY model             /app/model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
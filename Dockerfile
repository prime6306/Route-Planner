FROM python:3.11-slim

WORKDIR /app

# install deps first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# generate data and train models during build
# (this means the image ships with pre-trained models, no cold-start)
RUN python data/generate_data.py && python scripts/train.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

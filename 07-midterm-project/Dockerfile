FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving scripts and trained model artifacts
COPY predict.py serve.py pipeline.bin feature_defaults.json model_metadata.json ./

EXPOSE 9696

CMD ["python", "serve.py"]

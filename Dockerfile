FROM python:3.11-slim

WORKDIR /app

COPY ohc_runner.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#ENTRYPOINT ["python", "ohc_runner.py"]

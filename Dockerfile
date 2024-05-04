FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY data_txt/ data_txt
COPY src/ src

RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/app.py"]

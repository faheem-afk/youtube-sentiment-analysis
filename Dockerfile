FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY /flask_app/ /app/ 

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]


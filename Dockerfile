FROM python:3.10-slim
LABEL authors="marc"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY NearestNeighbors.pkl .

RUN mkdir -p /app/data
COPY data/data_weight.p /app/data/
COPY data/infos_movies.p /app/data/

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
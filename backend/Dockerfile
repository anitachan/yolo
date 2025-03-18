FROM python:3.10

LABEL maintainer="backenderp"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "yolo_api:app", "--host", "0.0.0.0", "--port", "8000"]

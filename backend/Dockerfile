FROM python:3.10

LABEL maintainer="backenderp"
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies; ensure requirements.txt is in the backend folder
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy backend code
COPY . .

EXPOSE 8000

# Run the FastAPI application (assumes your FastAPI app is in yolo_api.py with "app" defined)
CMD ["uvicorn", "yolo_api:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1.0.0-experimental
ARG BUILDPLATFORM=linux/amd64

# Base image with specified platform
FROM --platform=${BUILDPLATFORM} python:3.9-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./

EXPOSE 5002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5002"]

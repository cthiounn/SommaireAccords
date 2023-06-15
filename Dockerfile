# syntax = docker/dockerfile:1.2
FROM python:3.10

RUN apt-get update && apt install -y tesseract-ocr-all poppler-utils

COPY requirements.txt /tmp/

RUN --mount=type=cache,target=/var/cache/pip pip install --upgrade pip
RUN --mount=type=cache,target=/var/cache/pip pip install --upgrade torch
RUN --mount=type=cache,target=/var/cache/pip pip install --requirement /tmp/requirements.txt

RUN apt-get update && apt install -y libgl1

RUN pip install -U 'git+https://github.com/nikhilweee/iopath'

WORKDIR /app
# Copy all the files of this project inside the container
COPY . .

CMD ["streamlit", "run", "streamlit-api.py","--server.port", "3838"]
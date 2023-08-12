# syntax = docker/dockerfile:1.2
FROM python:3.10

WORKDIR /app
# Copy all the files of this project inside the container
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/stage_dares/model_final.pth -o model_final.pth 
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/stage_dares/config.yaml -o config.yaml 
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/stage_dares/train_230810_1/config.yaml -o config_dares.yaml 
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/stage_dares/train_230810_1/model_final.pth -o model_final_dares_1.pth 
RUN --mount=type=cache,target=/var/cache/ curl -SL https://minio.lab.sspcloud.fr/cthiounn2/stage_dares/train_230810_2/model_final.pth -o model_final_dares_2.pth 

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
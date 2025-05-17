FROM python:3.12-slim-bullseye

RUN apt-get update

ENV APP_HOME=/home/web/

RUN mkdir -p ${APP_HOME}

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY web.py .
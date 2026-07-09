FROM python:3.12-slim-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV APP_HOME=/home/app/

RUN mkdir -p ${APP_HOME}

WORKDIR ${APP_HOME}

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

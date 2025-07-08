
####################################################################################333
FROM node:18 as git_stage

WORKDIR /app
RUN apt-get update && apt-get install -y git && apt-get purge -y --auto-remove  && rm -rf /var/lib/apt/lists/*
ARG RELEASE_TAG="v0.0.1"
RUN git clone --branch v0.0.1 --single-branch https://github.com/ikarus1211/StitcherA.git
 
####################################################################################333


FROM python:3.11-slim as stitcher

# build-essential  libpq-dev libgl1 ffmpeg libsm6 libxext6

RUN apt-get update && apt-get install -y build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=git_stage app/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY --from=git_stage app/ ./

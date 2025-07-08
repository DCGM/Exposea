
####################################################################################333
FROM node:18 as git_stage

WORKDIR /app
RUN apt-get update && apt-get install -y git && apt-get purge -y --auto-remove  && rm -rf /var/lib/apt/lists/*
ARG RELEASE_TAG="v0.0.2"
RUN git clone --single-branch https://github.com/ikarus1211/StitcherA.git
 
####################################################################################333


FROM python:3.11-slim as stitcher

RUN apt-get update && apt-get install -y build-essential git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=git_stage /app/StitcherA/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --from=git_stage /app/StitcherA ./
RUN echo "Contents of /app:" && ls -la /app
COPY entrypoint-oneshot.sh /app/entrypoint-oneshot.sh
COPY entrypoint-service.sh /app/entrypoint-service.sh
RUN chmod +x /app/entrypoint-oneshot.sh /app/entrypoint-service.sh



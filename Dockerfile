FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

RUN mkdir chatbot
RUN mkdir chatbot/file
RUN mkdir chatbot/app
COPY ./app chatbot/app
COPY ./file chatbot/file
COPY ./requirements.txt chatbot

RUN apt update && apt upgrade -y
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN pip install -r chatbot/requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /chatbot
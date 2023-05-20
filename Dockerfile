FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN mkdir gpt_models
RUN mkdir gpt_models/file
RUN mkdir gpt_models/app
COPY ./app gpt_models/app
COPY ./file gpt_models/file

RUN apt update && apt upgrade -y
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install llama-cpp-python



WORKDIR /gpt_models


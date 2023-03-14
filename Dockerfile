FROM nvcr.io/nvidia/pytorch:22.12-py3
COPY . /app
WORKDIR /app
RUN apt update && apt install tmux -y
RUN pip install -r requirements.txt --no-cache-dir
RUN python train.py
EXPOSE 8888
# RUN uvicorn main:app --host 0.0.0.0 --port 8888 --reload # for some reason this doesn't allow to connect outside the docker
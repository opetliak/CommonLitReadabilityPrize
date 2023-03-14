docker build --tag nlp_reg -f Dockerfile .
docker run -it --rm --gpus all -p 8889:8889 --name nlp_reg nlp_reg:latest uvicorn main:app --host 0.0.0.0 --port 8889 --reload
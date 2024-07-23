FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY train.py /app/train.py

ENTRYPOINT ["python", "train.py"]
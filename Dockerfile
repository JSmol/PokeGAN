
# this is a google supplied pytorch image
FROM gcr.io/cloud-aiplatform/training/pytorch-gpu.1-4:latest

RUN pip install pandas google-cloud-storage

WORKDIR /root
# COPY data/* ./data/
COPY src/task.py ./src/task.py
COPY src/data.py ./src/data.py
COPY src/model.py ./src/model.py
COPY src/train.py ./src/train.py

ENTRYPOINT ["python3", "/root/src/task.py"]

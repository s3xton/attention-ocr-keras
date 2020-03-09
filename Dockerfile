FROM tensorflow/tensorflow:2.1.0-gpu-py3

ADD . /myapp
WORKDIR /myapp

RUN pip install -r requirements.txt

CMD python src/main.py --dataset_dir /data/fsns
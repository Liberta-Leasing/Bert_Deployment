FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip -y

RUN apt-get -y install git


ENV TZ=Europe/Paris

RUN pip3 install -q transformers   # here or dans requirements text SAME


RUN git clone https://github.com/Liberta-Leasing/Bert_deployement.git

RUN pip install -r Bert_deployement/requirements.txt

RUN rm Bert_deployement/requirements.txt

CMD ["python3", "Bert_Deployment/main.py"]

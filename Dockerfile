FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip -y

RUN apt-get -y install git


ENV TZ=Europe/Paris

RUN pip3 install -q transformers   # here or dans requirements text SAME
RUN pip3 install sklearn
RUN pip3 install -U scikit-learn # here I should install a recent version of sklearn in requirements and not install and upgrade


RUN git clone https://github.com/Liberta-Leasing/Bert_deployment.git
COPY model.pt Bert_deployment


RUN pip install -r Bert_deployment/requirements.txt

RUN rm Bert_deployment/requirements.txt

#CMD ["python3", "Bert_deployment/main.py"]

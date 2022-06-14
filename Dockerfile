# Step 1: Choose the base image
FROM ubuntu:20.04

# Step 2: Install/Update pip
RUN apt-get update && apt-get install python3 python3-pip -y

# Step 3: Install git
RUN apt-get -y install git

# Step 4: Setting an Environment Variable
ENV TZ=Europe/Paris

# Step 5: Intalling packages (Should be in the requirements.txt)
RUN pip3 install -q transformers   # here or dans requirements text SAME
RUN pip3 install sklearn
RUN pip3 install -U scikit-learn # here I should install a recent version of sklearn in requirements and not install and upgrade

# Step 6: Cloning the repo into the ./home folder
RUN cd ./home && git clone https://github.com/Liberta-Leasing/Bert_deployment.git

# Step 7: Copy the model in the /home/Bert_deployment folder.
COPY model.pt ./home/Bert_deployment

# Step 8: Install our requeriments
RUN pip install -r ./home/Bert_deployment/requirements.txt

# Step 9: Set ./home/Bert_deployment as the working directory
WORKDIR ./home/Bert_deployment

# Step 10: Execute the code
CMD ["python3", "main.py"]

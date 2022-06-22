FROM amazon/aws-lambda-python:3.9

WORKDIR /home/ubuntu

RUN yum clean all

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Paris

RUN yum update -y && yum install -y make curl wget sudo libtool clang git gcc-c++.x86_64 libgl1 libgl1-mesa-glx mesa-libGL ffmpeg libsm6 libxext6 poppler-utils

RUN yum install python3 python3-pip -y

WORKDIR "${LAMBDA_TASK_ROOT}"

# Step 5: Intalling packages (Should be in the requirements.txt)
RUN pip3 install -q transformers --target "${LAMBDA_TASK_ROOT}"  # here or dans requirements text SAME
RUN pip3 install sklearn --target "${LAMBDA_TASK_ROOT}"
RUN pip3 install -U scikit-learn --target "${LAMBDA_TASK_ROOT}"# here I should install a recent version of sklearn in requirements and not install and upgrade

# Step 6: Cloning the repo into the ./home folder
RUN cd ./home && git clone https://github.com/Liberta-Leasing/Bert_deployment.git

# Step 7: Copy the model in the /home/Bert_deployment folder.
COPY model.pt "${LAMBDA_TASK_ROOT}"

# Step 8: Install our requeriments
RUN pip install -r ./home/Bert_deployment/requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Step 9: Set ./home/Bert_deployment as the working directory
WORKDIR "${LAMBDA_TASK_ROOT}"

# Step 10: Execute the code
CMD ["main.lambda_handler"]

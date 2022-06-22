FROM amazon/aws-lambda-python:3.9

WORKDIR /home/ubuntu

RUN yum clean all

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Paris

RUN yum update -y && yum install -y make curl wget sudo libtool clang git gcc-c++.x86_64 libgl1 libgl1-mesa-glx mesa-libGL ffmpeg libsm6 libxext6 poppler-utils

RUN yum install python3 python3-pip -y

WORKDIR "${LAMBDA_TASK_ROOT}"

# Step 6: Cloning the repo into the ./home folder
RUN git clone https://github.com/Liberta-Leasing/Bert_deployment.git

# Step 7: Copy the model in the /home/Bert_deployment folder.
#COPY model.pt "${LAMBDA_TASK_ROOT}"

# Step 8: Install our requeriments
RUN pip install -r Bert_deployment/requirements.txt --target "${LAMBDA_TASK_ROOT}"

RUN cd ./Bert_deployment

# Step 9: Set ./home/Bert_deployment as the working directory
WORKDIR "${LAMBDA_TASK_ROOT}"

# Step 10: Execute the code
CMD ["main.lambda_handler"]

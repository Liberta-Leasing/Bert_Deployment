from predict import load_model
from predict import classifier
import boto3
import sys

sys.path.insert(0, '/tmp/')

import botocore
import os
import subprocess
import json


s3 = boto3.resource('s3')

def lambda_handler(event, context):

    print(event)
    #download the image
    csv_i = event["Records"][0]["s3"]["object"]["key"]
    csv_key = event["Records"][0]["s3"]["object"]["key"].split('/')
    filename = csv_key[-1]
    local_file = '/tmp/'+filename
    download_from_s3(csv_i,local_file)

    # All the necessary steps to execute the model
    model_path = s3.Bucket('bert-weights').download_file('model.pt','/tmp/model.pt')

    # Loading the model
    loaded_model=load_model(model_path)
    # Predict

    classifier(local_file, loaded_model)

    return {
        'statusCode': 200,
        'body': json.dumps({"output":"Lambda execution was successful"})
    }
  
    # Predict

def download_from_s3(file,object_name):
    try:
        s3.Bucket('statementsoutput').download_file(file,object_name)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise



#Function to upload files to s3
def upload_file(file_name, object_name=None):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # Upload the file
    try:
        response = s3_client.upload_file(file_name, 'statementsoutput', object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
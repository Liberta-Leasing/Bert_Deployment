#from predict import load_model
#from predict import classifier
import boto3
from botocore.exceptions import ClientError
import sys

sys.path.insert(0, '/tmp/')

import botocore
import os
import subprocess
import json
import glob

import torch
import csv
from zipfile import ZipFile

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast, AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

def lambda_handler(event, context):

    os.makedirs('/tmp/tesseract_csv_zip', exist_ok= True) 
    print("1 : /tmp/tesseract_csv_zip was created")

    os.makedirs('/tmp/bert_output_zip', exist_ok= True) 
    print("1 : /tmp/bert_output_zip was created")

    os.makedirs('/tmp/bert_output', exist_ok= True) 
    print("1 : /tmp/bert_output was created")

    print(event)
    #download the image
    csv_key = event["Records"][0]["s3"]["object"]["key"]
    print(csv_key)
    csv_split = event["Records"][0]["s3"]["object"]["key"].split('/')
    input_csv = csv_split[-1]
    print(input_csv)
    folder = csv_split[1]
    local_zip_file = f'/tmp/{input_csv}'
    #download_from_s3(filename, local_file)
    download_from_s3(csv_key, local_zip_file)
    list_of_files = unzip(local_zip_file)

    output_files = []
    #print("####################\n")  
    #print("WHERE ARE MY FILES?\n")
    #print("####################\n")  
    #print("step 1: listing files matching /tmp/*.csv ")
    #print(glob.glob("/tmp/*.csv"))
    #print("step 2: listing the files matching tmp/*.csv")
    #print(glob.glob("tmp/*.csv"))
    #print("step 3: listing the files in tmp/")
    #print(os.listdir("/tmp"))
    #print("step 4: listing the files matching /tmp/tesseract_csv_zip/*.csv" )
    #print(glob.glob("/tmp/tmp/tesseract_csv_zip/*.csv"))
    #print("step 5: listing the files matching /tmp/tesseract_csv_zip")
    #print(os.listdir("/tmp/tesseract_csv_zip"))
    #print("step 6: listing the files matching /tmp/tesseract_csv_zip/tmp")
    #print(os.listdir("/tmp/tesseract_csv_zip/tmp"))
    #print("step 7: listing the files matching /tmp/tesseract_csv_zip/tmp/yolo_output_zip/")
    #print(os.listdir("/tmp/tesseract_csv_zip/tmp/yolo_output_zip"))

    print("####################\n")  
    print("BERT\n")
    print("####################\n") 
  # Goes through all images in the folder.
    for csv_file in glob.glob("/tmp/tesseract_csv_zip/tmp/yolo_output_zip/*.csv"):
      try:
        #print("toto")

        # All the necessary steps to execute the model
        model_path = s3.Bucket('bert-weights').download_file('model.pt','/tmp/model.pt')

        # Loading the model
        loaded_model=load_model('/tmp/model.pt')        
        # Predict

        classifier(csv_file, loaded_model)

        output_files.append(f'/tmp/bert_output/{csv_file[:-4]}.csv')

      except Exception as e :
        print("error for csv_file : ", csv_file)
        print(e)
        continue

    print(output_files)
    zip_files(output_files)
    
    print("####################\n")  
    print("OUTPUT\n")
    print("####################\n") 
    # will be problematic if we want to keep track of the customer 
    upload_file('/tmp/bert_csv.zip','processing/bert_output/bert_csv.zip')

    return "output: Lambda execution was successful"
  
    # Predict

# DOCSTRINGS PLEASE
def zip_files(files):
    # writing files to a zipfile
    with ZipFile('/tmp/bert_csv.zip','w') as zip:
        # writing each file one by one
        for file in files:
            zip.write(file) 


def unzip(zipped_file_name):
    directory = os.getcwd()
    os.chdir('/tmp/tesseract_csv_zip')
    # opening the zip file in READ mode
    with ZipFile(zipped_file_name, 'r') as zip:
        # printing all the contents of the zip file
        zip.printdir()
        list_of_files = zip.namelist()
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall()
        print('Unzipping Done!')
        os.chdir(directory)
        return list_of_files   


def download_from_s3(remoteFilename,local_file):
    try:
        s3.Bucket('statementsoutput').download_file(remoteFilename,local_file)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def load_model(model_path):
	
	loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
	return loaded_model

# we have to add the loaded_model import here or another script
def classifier(local_file, loaded_model): # not possible when using lambda function 
   
  df = pd.read_csv(local_file, sep =',' )
  transaction_list = [] 
  pred_list= []
  pred_code_list=[]
  for index, row in df.iterrows():
    text = row['DESCRIPTION']
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="multi_label_classification",cache_dir="/tmp")
    encoding = tokenizer(text, return_tensors="pt")#
    outputs = loaded_model(**encoding)
    predictions = outputs.logits.argmax(-1)
    int(predictions)
    transaction_list.append(text)
    pred_code_list.append(int(predictions))
    if int(predictions) == 0:
      pred_list.append('TRANSFER')
    elif int(predictions)== 1:
      pred_list.append('PURCHASE')
    elif int(predictions) == 2:
      pred_list.append('LOAN')
    elif int(predictions)== 3:
      pred_list.append('CHARGES')
    elif int(predictions)== 4:
      pred_list.append('SALARY')
    elif int(predictions) == 5:
      pred_list.append('CASH')
    elif int(predictions)== 6:
      pred_list.append('REVERSAL')
    elif int(predictions)== 7:
      pred_list.append('CHEQUE')
    elif int(predictions)== 8:
      pred_list.append('PAYMENT')
    elif int(predictions)== 9:
      pred_list.append('UNKNOWN')
     
    #print(row['BANK_ID'],  '|' , 'The transcation '+ '"' + (row['DESCRIPTION']) + 
      #  '"', 'corresponds to the category ' , int(predictions))

    print( 'The transcation '+ '"' + (row['DESCRIPTION']) + 
        '"', 'corresponds to the category ' , int(predictions))
    
  df1 = pd.DataFrame(list(zip(transaction_list, pred_list, pred_code_list)), columns = ['TRANSACTION', 'CATEGORY', 'CATEGORY_CODE'])

  df1.to_csv(f'/tmp/bert_output/{local_file[:-4]}.csv')


#Function to upload files to s3
def upload_file(df_csv, object_name):
    """Upload a file to an S3 bucket
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # Upload the file
    try:
        response = s3_client.upload_file(df_csv, 'statementsoutput', object_name)
    except ClientError as e:
        print("Upload failed")
        #logging.error(e)
        return False
    return True
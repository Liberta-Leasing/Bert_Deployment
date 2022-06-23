# script for predict
import torch
import csv

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

def load_model(model_path):
	
	loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
	return loaded_model

# we have to add the loaded_model import here or another script
def classifier(INPUT_FILE, loaded_model): # not possible when using lambda function 
   
  df = pd.read_csv(local_file, sep =';' )
  transaction_list = [] 
  pred_list= []
  pred_code_list=[]
  for index, row in df.iterrows():
    text = row['DESCRIPTION']
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, problem_type="multi_label_classification")
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



     
    print(row['BANK_ID'],  '|' , 'The transcation '+ '"' + (row['DESCRIPTION']) + 
        '"', 'corresponds to the category ' , int(predictions))
    
  df1 = pd.DataFrame(list(zip(transaction_list, pred_list, pred_code_list)), columns = ['TRANSACTION', 'CATEGORY', 'CATEGORY_CODE'])

  df1.to_csv('my_ouptput.csv')



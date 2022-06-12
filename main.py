from to_load import load_model
import config
from predict import *

def main():

    # All the necessary steps to execute the model

    # Loading the model
    loaded_model=load_model(config.MODEL_PATH)
    # Predict

    classifier(config.INPUT_FILE, loaded_model)
  
    # Predict
  
  
  
# to test it locally
if __name__ == "__main__":
    main()
import torch 
import config


# to used as a module function

def load_model(model_path):
	
	loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
	return loaded_model
	

# to test it locally
if __name__ == "__main__":

	model_path = config.MODEL_PATH
	print(model_path)
	model = load_model(model_path)
	model.eval()

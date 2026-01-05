import torch
import numpy as np
import json
import random
from pathlib import Path
def seed_everything(seed=42):
 random.seed(seed)
 np.random.seed(seed)
 torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True
def load_config(config_path):
 with open(config_path,'r') as f:
  config=json.load(f)
 return config
def save_config(config,config_path):
 Path(config_path).parent.mkdir(parents=True,exist_ok=True)
 with open(config_path,'w') as f:
  json.dump(config,f,indent=2)
def save_model(model,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 torch.save(model.state_dict(),path)
def load_model(model,path,device='cpu'):
 model.load_state_dict(torch.load(path,map_location=device))
 return model
def save_metrics(metrics,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with open(path,'w') as f:
  json.dump(metrics,f,indent=2)
def log_metrics(metrics,output_path):
 with open(output_path,'a') as f:
  f.write(json.dumps(metrics)+'\n')
class EarlyStopping:
 def __init__(self,patience=10,min_delta=0.0,mode='min'):
  self.patience=patience
  self.min_delta=min_delta
  self.mode=mode
  self.best_value=None
  self.counter=0
  self.early_stop=False
 def __call__(self,value):
  if self.best_value is None:
   self.best_value=value
  elif self.mode=='min':
   if value<self.best_value-self.min_delta:
    self.best_value=value
    self.counter=0
   else:
    self.counter+=1
  else:
   if value>self.best_value+self.min_delta:
    self.best_value=value
    self.counter=0
   else:
    self.counter+=1
  if self.counter>=self.patience:
   self.early_stop=True
  return self.early_stop
def get_device(use_cuda=True):
 if use_cuda and torch.cuda.is_available():
  return torch.device('cuda')
 return torch.device('cpu')
def count_parameters(model):
 return sum(p.numel() for p in model.parameters() if p.requires_grad)
def freeze_encoder(model):
 for param in model.encoder.parameters():
  param.requires_grad=False
def unfreeze_encoder(model):
 for param in model.encoder.parameters():
  param.requires_grad=True

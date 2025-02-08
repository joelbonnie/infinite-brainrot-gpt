
# Importing libraries 

import torch
import torch.nn as nn
from nn import functional as F



####################
## Hyper-params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
block_size = 256

alpha = 3e-4

train_iters = 4000
eval_iters = 200
eval_interval = 500

embed_dim = 384

head_num = 6
block_num = 6
dropout = 0.2

####################


####################
## Constants 

dataset_path = ""
weights_path = ""

####################



####################
## Model 

torch.manual_seed(42)


# Self-Attention Head 
class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super.__init__()

    def forward(self, x):
        return
 

# Multi-Headed Attention 
class MultiAttentionHead(nn.Module):

    def __init__(self, head_num, head_size):
        super.__init__()

    def forward(self, x):
        return 

# ReLU Feed-Forward 
class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super.__init__()

    def forward(self, x):
        return


# Model Block 
class ModelBlock(nn.Module):

    def __init__(self, head_num, embed_dim):
        super.__init__()

    def forward(self, x):
        return

# GPT
class BrainRotGPT(nn.Module):

    def __init__(self):
        super.__init__()

    def forward(self, input_arr, targets=None):
        return

    def generate(self, input_arr, max_tokens):
        return input_arr 


####################




####################
## Data


# Dataset Load in
with open(dataset_path, 'r', encoding='utf-8') as fd:
    input_text = fd.read()

# Tokenize
vocab = sorted(list(set(input_text)))
str_to_tokens = {}
tokens_to_str= {}

def encode(s):
    return

def decode(t):  
    return

# Train-Test Split 


# Loss Function 
def loss_fn():
    return 

####################


# Training Generation 





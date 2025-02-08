
# Importing libraries 

import torch
import torch.nn as nn
from torch.nn import functional as F



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

dataset_path = "input.txt"
weights_path = "weights.pth"

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
print(vocab)
str_to_tokens = {c:i for i,c in enumerate(vocab)}
tokens_to_str= {i:c for i,c in enumerate(vocab)}

def encode(s):
    return [str_to_tokens[c] for c in s]

def decode(t):  
    return ''.join([tokens_to_str[i] for i in t])


# Train-Test Split 
text_tensor = torch.tensor(encode(input_text), dtype=torch.long)
split_num = int(0.9*len(text_tensor))

train_text = text_tensor[:split_num]
test_text = text_tensor[split_num:]

# Get Batches
def get_batch(batch_type):
    curr_data = train_text if batch_type == 'train' else test_text
    indexes = torch.randint(len(curr_data) - batch_size, (batch_size,))


# Loss Function 
@torch.no_grad()
def loss_fn():
    return 

####################


# Training Generation 





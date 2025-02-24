
# Importing libraries 

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sentencepiece as spm


####################
## Hyper-params

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
block_size = 32

alpha = 3e-4

train_iters = 6000
eval_iters = 200
eval_interval = 500

embed_dim = 128

head_num = 6
block_num = 6
dropout = 0.2

####################


####################
## Constants 

dataset_path = "text/processed_brainrot_trimmed.txt"
weights_path = "weights/weights_unigram.pth"
tokenizer_model_path = "token/brainrot_tokenizer.model"

####################



####################
## Model 

#torch.manual_seed(42)


# Self-Attention Head 
class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        # K, Q, V

        self.key = nn.Linear(embed_dim, head_size, bias = False)
        self.query = nn.Linear(embed_dim, head_size, bias = False)
        self.value = nn.Linear(embed_dim, head_size, bias = False)
        
        # Decoder 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Affinities 
        weights = queries @ keys.transpose(-2,-1)
        weights = weights * keys.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ values
        return output
            
 

# Multi-Headed Attention 
class MultiAttentionHead(nn.Module):

    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(head_num)])
        self.proj = nn.Linear(head_num * head_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


# ReLU Feed-Forward 
class FeedForward(nn.Module):

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Model Block 
class ModelBlock(nn.Module):

    def __init__(self, head_num, embed_dim):
        super().__init__()
        head_size = embed_dim // head_num

        self.attention = MultiAttentionHead(head_num, head_size)
        self.feedfoward = FeedForward(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = x + self.attention(self.layernorm1(x))
        x = x + self.feedfoward(self.layernorm2(x))
        return x 

# GPT
class BrainRotGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding_table = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[ModelBlock(head_num,embed_dim) for _ in range(block_num)])
        self.layernormfinal= nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size) 

        # initialize weights
        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, input_arr, targets=None):
        B,T = input_arr.shape
        
        # Embedding Tables
        token_embeds= self.token_embedding_table(input_arr)
        position_embeds = self.pos_embedding_table(torch.arange(T, device=device))

        x = token_embeds + position_embeds

        x = self.blocks(x)
        x = self.layernormfinal(x)

        logits = self.linear(x)

        if targets is None:
            # No loss
            loss = None
        else: 
            B, T, C = logits.shape

            # reshape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss



    def generate(self, input_arr, max_tokens):
        for _ in range(max_tokens):
            input_end = input_arr[:, -block_size:]
            logits, loss = self(input_end)
            logits_last_time = logits[:,-1,:]
            probs = F.softmax(logits_last_time, dim=-1)
            next_char = torch.multinomial(probs, num_samples=1)
            input_arr = torch.cat((input_arr, next_char), dim=1)

        return input_arr 


####################




####################
## Data


# Dataset Load in
with open(dataset_path, 'r', encoding='utf-8') as fd:
    input_text = fd.read()

# Tokenize
"""
vocab = sorted(list(set(input_text)))
vocab_size = len(vocab)
print(vocab)
str_to_tokens = {c:i for i,c in enumerate(vocab)}
tokens_to_str= {i:c for i,c in enumerate(vocab)}

def encode(s):
    return [str_to_tokens[c] for c in s]

def decode(t):  
    return ''.join([tokens_to_str[i] for i in t])
"""

sp = spm.SentencePieceProcessor()
sp.load(tokenizer_model_path)

def encode(text):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)

vocab_size = sp.get_piece_size()


# Train-Test Split 
text_tensor = torch.tensor(encode(input_text), dtype=torch.long, device=device)
split_num = int(0.9*len(text_tensor))

train_text = text_tensor[:split_num]
test_text = text_tensor[split_num:]

# Get Batches
def get_batch(batch_type):
    curr_data = train_text if batch_type == 'train' else test_text
    indexes = torch.randint(len(curr_data) - block_size, (batch_size,), device=device)
    x = torch.stack([curr_data[i:i+block_size] for i in indexes]).to(device)
    y = torch.stack([curr_data[i+1:i+block_size+1] for i in indexes]).to(device)
    return x,y


# Loss Function 
@torch.no_grad()
def loss_fn():
    output = {}
    gpt_model.eval()
    for split_type in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            X, Y = get_batch(split_type)
            logits, loss = gpt_model(X,Y)
            losses[iter] = loss.item()
        output[split_type] = losses.mean()
    gpt_model.train()
    return output

####################


# Training Generation 
gpt_model = BrainRotGPT().to(device)
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=alpha)


if weights_path != "" and os.path.exists(weights_path):
    weights = torch.load(weights_path, map_location=device)
    gpt_model.load_state_dict(weights['model_state_dict'])
    optimizer.load_state_dict(weights['optimizer_state_dict'])
    print("Weights Loaded")

else:
    # Training
    # Check pre-trained weights
    for curr_iter in range(train_iters):
        print(f"Training: Epoch {curr_iter}")
    
        # evaluate 
        if curr_iter == train_iters - 1 or curr_iter % eval_interval == 0:
            losses = loss_fn()
            print(f"Training: Epoch {curr_iter} Evaluation: Training {losses['train']}, Test {losses['test']}")

        x_b, y_b = get_batch('train')

        logits, loss = gpt_model(x_b, y_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if weights_path != "":
        torch.save({
            'model_state_dict': gpt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, weights_path)
        print("Weights Saved")

# Text Generation 
context = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(gpt_model.generate(context, max_tokens=100)[0].tolist()))



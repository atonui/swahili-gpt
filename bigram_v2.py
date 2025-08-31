import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many indpendent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ability to run on a GPU if you have it
eval_iters = 200
n_embed = 32 # number of embedding dimensions

torch.manual_seed(1337) # make the code reproduceable when you run it again

# load the training dataset
with open('train.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# load the test dataset
with open('test.txt', 'r', encoding='utf-8') as f:
    test = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenisation by mapping characters to integers and vice versa
stoi = {ch:i for i, ch in enumerate(chars)} # string to integer
itoi = {i:ch for i, ch in enumerate(chars)} # integer to string
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string and outputs a list of integers
decode = lambda l: ''.join([itoi[i] for i in l]) # decoder: takes a list of integers and outputs a string

# create the data tensors
train_data = torch.tensor(encode(text), dtype=torch.long)
val_data = torch.tensor(encode(test), dtype=torch.long)

# loading the data
def get_batch(split):
    '''generate a small batch of data of inputs and targets y'''
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''one head of self attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores i.e. affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T) ---> future does not communicate with the past
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # language modeling head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embeddings = self.token_embedding_table(idx) # (Batch,Time,Channel) tensor
        positional_embeddings = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = token_embeddings + positional_embeddings
        x = self.sa_head(x) # apply one head of self attention (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size), probability of next token

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # print('idx_cond: ',idx_cond)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # print('idx_next: ',idx)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        #evaluate the loss
        logits, loss = model(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
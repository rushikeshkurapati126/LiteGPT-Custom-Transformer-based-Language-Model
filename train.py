import torch
import torch.nn as nn
import os
import pickle

# =========================
# DEVICE
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# FILE PATH
# =========================
file_path = r"C:\Users\rushi\OneDrive\Desktop\llm\llm dataset\pg100.txt"

if not os.path.exists(file_path):
    print("❌ File not found")
    exit()

# =========================
# LOAD DATA
# =========================
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# =========================
# TOKENIZATION
# =========================
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# =========================
# SAVE VOCAB (IMPORTANT)
# =========================
with open("vocab.pkl", "wb") as f:
    pickle.dump((stoi, itos), f)

print("✅ Vocab saved")

# =========================
# SPLIT
# =========================
n = int(0.9 * len(data))
train_data = data[:n]

# =========================
# PARAMETERS
# =========================
block_size = 64
batch_size = 16
n_embd = 128

# =========================
# BATCH
# =========================
def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# =========================
# MODEL
# =========================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size)).to(device)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd // 4) for _ in range(4)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(Block(), Block())
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        B,T = x.shape
        x = self.token_embed(x) + self.pos_embed(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            loss = nn.CrossEntropyLoss()(logits, y)

        return logits, loss

# =========================
# TRAIN
# =========================
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(3000):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"Step {step}, Loss: {loss.item()}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved")
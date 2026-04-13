import torch
import torch.nn as nn
import pickle

# =========================
# DEVICE
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# LOAD VOCAB (IMPORTANT)
# =========================
with open("vocab.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

vocab_size = len(stoi)

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# =========================
# PARAMETERS
# =========================
block_size = 64
n_embd = 128

# =========================
# MODEL (same as training)
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

    def forward(self, x):
        B,T = x.shape
        x = self.token_embed(x) + self.pos_embed(torch.arange(T, device=device))
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)

# =========================
# LOAD MODEL
# =========================
model = MiniGPT().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

print("✅ Chat ready (type 'exit' to quit)\n")

# =========================
# CHAT LOOP
# =========================
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    context = torch.tensor([encode(user_input)], dtype=torch.long).to(device)

    for _ in range(200):
        context_cond = context[:, -block_size:]
        logits = model(context_cond)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

    print("Bot:", decode(context[0].tolist()))
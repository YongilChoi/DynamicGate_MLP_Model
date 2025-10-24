# ============================================
# DynamicGate-MLP MNIST 학습 + 게이트 모니터링/CSV 저장
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Gate Layer ----------------
class GateLayer(nn.Module):
    def __init__(self, size, tau=0.5, init_val=2.0):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.ones(size) * init_val)
        self.tau = tau

    def forward(self, x):
        probs = torch.sigmoid(self.gate_logits)
        hard = (probs > self.tau).float()
        gates = hard + (probs - hard).detach()   # STE
        return x * gates, probs

# ---------------- Model ---------------------
class DynamicGateMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, tau=0.5):
        super().__init__()
        self.gate_in  = GateLayer(input_dim, tau)
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.gate_hid = GateLayer(hidden_dim, tau)
        self.fc2      = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x, probs_in = self.gate_in(x)
        x = self.fc1(x)
        x, probs_hid = self.gate_hid(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits, (probs_in, probs_hid)

# ---------------- Loss 함수 -----------------
def gate_loss(logits, targets, probs_in, probs_hid, beta=1e-3):
    ce = F.cross_entropy(logits, targets)
    reg = (probs_in.mean() + probs_hid.mean()) / 2.0
    return ce + beta * reg

# ---------------- 게이트 로깅 클래스 -----------------
class GateLogger:
    def __init__(self):
        self.logs = []

    def log(self, epoch, model, acc):
        with torch.no_grad():
            p_in = torch.sigmoid(model.gate_in.gate_logits).mean().item()
            h_in = (torch.sigmoid(model.gate_in.gate_logits) > model.gate_in.tau).float().mean().item()
            p_hid = torch.sigmoid(model.gate_hid.gate_logits).mean().item()
            h_hid = (torch.sigmoid(model.gate_hid.gate_logits) > model.gate_hid.tau).float().mean().item()
        self.logs.append({
            "epoch": epoch,
            "accuracy": acc,
            "mean_prob_in": p_in,
            "active_ratio_in": h_in,
            "mean_prob_hid": p_hid,
            "active_ratio_hid": h_hid
        })

    def to_dataframe(self):
        return pd.DataFrame(self.logs)

# ---------------- 데이터 준비 -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

# ---------------- 학습 -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DynamicGateMLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

gate_logger = GateLogger()
epochs = 5
for epoch in range(1, epochs+1):
    # --- Train ---
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits, (p_in, p_hid) = model(xb)
        loss = gate_loss(logits, yb, p_in, p_hid, beta=1e-3)
        loss.backward()
        opt.step()

    # --- Eval ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    print(f"Epoch {epoch}/{epochs} - Test Accuracy: {acc*100:.2f}%")
    gate_logger.log(epoch, model, acc)

# ---------------- CSV 저장 + 시각화 -----------------
df = gate_logger.to_dataframe()
df.to_csv("gate_stats.csv", index=False)
print(df)

# 정확도 & 게이트 확률 변화 그래프
plt.figure(figsize=(10,4))
plt.plot(df["epoch"], df["accuracy"], label="Accuracy")
plt.plot(df["epoch"], df["mean_prob_in"], label="Mean Gate Prob (Input)")
plt.plot(df["epoch"], df["mean_prob_hid"], label="Mean Gate Prob (Hidden)")
plt.xlabel("Epoch")
plt.legend()
plt.title("Accuracy & Gate Probability over Epochs")
plt.show()


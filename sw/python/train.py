from pathlib import Path

import torch
import numpy as np
import torch.nn as nn

# paths are relative to the repo root so the script runs from any folder
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "weights" / "fer3_float.npz"

torch.manual_seed(0) #same seed -> same weights every run

d = np.load(ROOT / "fer3.npz")

# (x - 128) / 128 puts pixels in [-1, 1). On the FPGA this is just x - 128 as an int8.
x_train = torch.from_numpy((d["x_train"].astype(np.float32) - 128) / 128)
y_train = torch.from_numpy(d["y_train"])
x_val = torch.from_numpy((d["x_val"].astype(np.float32) - 128) / 128)
y_val = torch.from_numpy(d["y_val"])

model = nn.Sequential(
    #hidden layer: 2304 pixels -> 64 neurons
    nn.Linear(2304, 64),
    nn.ReLU(),

    #output layer: 64 neurons -> 3 class scores
    nn.Linear(64, 3),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

best_val_acc = 0.0
best_state = None

for epoch in range(20):
    #reshuffle every epoch so batches are different each time
    order = torch.randperm(len(x_train))
    total_loss = 0.0
    for i in range(0, len(x_train), 64):
        idx = order[i:i+64]
        xb = x_train[idx]
        yb = y_train[idx]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)

    with torch.no_grad():
        train_acc = (model(x_train).argmax(1) == y_train).float().mean().item()
        val_acc = (model(x_val).argmax(1) == y_val).float().mean().item()

    #keep the weights from the epoch that did best on validation, not the last one
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        marker = "  <- best"
    print(f"{epoch:2d}  loss {total_loss / len(x_train):.4f}  "
          f"train {train_acc:.4f}  val {val_acc:.4f}{marker}")

# the test set is deliberately not touched here -- quantize.py reports it once at the end

# save as plain numpy arrays so quantize.py doesn't need torch
np.savez(
    OUT,
    w1=best_state["0.weight"].numpy(),   # (64, 2304)
    b1=best_state["0.bias"].numpy(),     # (64,)
    w2=best_state["2.weight"].numpy(),   # (3, 64)
    b2=best_state["2.bias"].numpy(),     # (3,)
    best_val_acc=best_val_acc,
)
print(f"best val acc {best_val_acc:.4f}, saved to {OUT.relative_to(ROOT)}")

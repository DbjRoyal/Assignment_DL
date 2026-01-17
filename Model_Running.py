import numpy as np
import torch.nn as nn

n_classes = 5  # 0–4 (exclude 255)

class_counts = np.array([(Y_train == i).sum() for i in range(n_classes)])
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32).to(device),
    ignore_index=255
)

train_loss, val_loss, val_acc = [], [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    tl = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        tl += loss.item()

    train_loss.append(tl / len(train_loader))

    # Validation
    model.eval()
    correct, total, vl = 0, 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            vl += criterion(out, yb).item()

            pred = out.argmax(1)
            mask = yb != 255
            correct += (pred[mask] == yb[mask]).sum().item()
            total += mask.sum().item()

    val_loss.append(vl / len(val_loader))
    val_acc.append(correct / total)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss[-1]:.4f} | "
          f"Val Loss: {val_loss[-1]:.4f} | "
          f"Val Acc: {val_acc[-1]:.4f}")

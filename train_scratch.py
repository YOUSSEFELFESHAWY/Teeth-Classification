import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
DATA_ROOT = "Teeth DataSet/Teeth_Dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "Training")
VAL_DIR   = os.path.join(DATA_ROOT, "Validation")

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_BEST_PATH = os.path.join(OUTPUT_DIR, "scratch_cnn_best.pth")
MODEL_LAST_PATH = os.path.join(OUTPUT_DIR, "scratch_cnn_last.pth")
CLASSES_PATH = os.path.join(OUTPUT_DIR, "classes.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# TRANSFORMS
# =========================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# DATA
# =========================
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_data   = datasets.ImageFolder(VAL_DIR, transform=transform_val)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

classes = train_data.classes
num_classes = len(classes)

# Save classes once (important for evaluation)
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    for c in classes:
        f.write(c + "\n")
print("Classes:", classes)
print(f"✅ Classes saved to: {CLASSES_PATH}")

# =========================
# SCRATCH CNN
# =========================
class ScratchCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

model = ScratchCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# EVAL FUNCTION
# =========================
def evaluate():
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item()
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss = loss_sum / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

# =========================
# TRAIN LOOP
# =========================
best_acc = -1.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()

    train_loss = running / len(train_loader)
    val_loss, val_acc = evaluate()

    print(f"Epoch {epoch}/{EPOCHS} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}%")

    # Save best
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_BEST_PATH)
        print(f"✅ Best scratch model saved: {MODEL_BEST_PATH} (ValAcc={best_acc:.2f}%)")

# Save last
torch.save(model.state_dict(), MODEL_LAST_PATH)
print(f"✅ Last scratch model saved: {MODEL_LAST_PATH}")

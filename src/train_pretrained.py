import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATA_ROOT = "Teeth DataSet/Teeth_Dataset"
TRAIN_DIR = f"{DATA_ROOT}/Training"
VAL_DIR   = f"{DATA_ROOT}/Validation"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LAST_PATH = os.path.join(OUTPUT_DIR, "pretrained_resnet18_last.pth")
MODEL_BEST_PATH = os.path.join(OUTPUT_DIR, "pretrained_resnet18_best.pth")
CLASSES_PATH    = os.path.join(OUTPUT_DIR, "classes.txt")

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

train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_data   = datasets.ImageFolder(VAL_DIR, transform=transform_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)

# Save classes order (important for evaluation)
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    for c in train_data.classes:
        f.write(c + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Classes:", train_data.classes)

# Pretrained model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
    return loss_sum / len(val_loader), 100 * correct / total

epochs = 3
best_acc = -1.0

for epoch in range(1, epochs + 1):
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

    val_loss, val_acc = evaluate()
    print(f"Epoch {epoch}/{epochs} | TrainLoss: {running/len(train_loader):.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_BEST_PATH)
        print(f"✅ Best model saved: {MODEL_BEST_PATH} (ValAcc={best_acc:.2f}%)")

# Save last model after training
torch.save(model.state_dict(), MODEL_LAST_PATH)
print(f"✅ Last model saved: {MODEL_LAST_PATH}")
print(f"✅ Classes saved: {CLASSES_PATH}")
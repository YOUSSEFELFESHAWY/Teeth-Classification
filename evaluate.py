import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# =========================
# CONFIG
# =========================
DATA_ROOT = "Teeth DataSet/Teeth_Dataset"
EVAL_SPLIT = "Validation"  # "Validation" or "Testing"
BATCH_SIZE = 16

# اختر النوع هنا:
MODEL_TYPE = "scratch"  # "pretrained" or "scratch"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATHS = {
    "pretrained": os.path.join(OUTPUT_DIR, "pretrained_resnet18_best.pth"),
    "scratch": os.path.join(OUTPUT_DIR, "scratch_cnn_best.pth"),
}

MODEL_PATH = MODEL_PATHS[MODEL_TYPE]

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# DATA
# =========================
eval_data = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, EVAL_SPLIT),
    transform=transform
)

eval_loader = DataLoader(eval_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = eval_data.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Classes:", class_names)

# =========================
# MODEL
# =========================
if MODEL_TYPE == "pretrained":
    model = models.resnet18(weights=None)  # مهم: weights=None لأننا هنحمّل weights من .pth
    model.fc = nn.Linear(model.fc.in_features, num_classes)
else:
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

    model = ScratchCNN(num_classes)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}\nRun training first to generate it.")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# EVALUATION
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================
# METRICS
# =========================
acc = (all_preds == all_labels).mean() * 100
print(f"\n✅ Evaluation Accuracy ({MODEL_TYPE}) on {EVAL_SPLIT}: {acc:.2f}%\n")

report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

report_path = os.path.join(OUTPUT_DIR, f"classification_report_{MODEL_TYPE}_{EVAL_SPLIT.lower()}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

# Save json metrics (for comparison)
metrics = {
    "model_type": MODEL_TYPE,
    "eval_split": EVAL_SPLIT,
    "accuracy": float(acc),
}
metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{MODEL_TYPE}_{EVAL_SPLIT.lower()}.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix ({MODEL_TYPE} - {EVAL_SPLIT})")
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{MODEL_TYPE}_{EVAL_SPLIT.lower()}.png")
plt.savefig(cm_path)
plt.show()

print(f"✅ Confusion matrix saved to: {cm_path}")
print(f"✅ Report saved to: {report_path}")
print(f"✅ Metrics saved to: {metrics_path}")

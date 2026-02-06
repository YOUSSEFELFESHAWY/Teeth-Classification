import os
import time
import json
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# =========================
# CONFIG
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

# نفس الداتا بتاعتك (زي اللي في السكربتات)
DATA_ROOT = os.path.join(PROJECT_ROOT, "Teeth DataSet", "Teeth_Dataset")

CLASSES_PATH = os.path.join(OUTPUT_DIR, "classes.txt")

# Paths متوقعة للـ models (عدّل الاسم لو مختلف عندك)
PRETRAINED_BEST = os.path.join(OUTPUT_DIR, "pretrained_resnet18_best.pth")
PRETRAINED_LAST = os.path.join(OUTPUT_DIR, "pretrained_resnet18_last.pth")

SCRATCH_BEST    = os.path.join(OUTPUT_DIR, "scratch_cnn_best.pth")
SCRATCH_LAST    = os.path.join(OUTPUT_DIR, "scratch_cnn_last.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# UI SETTINGS
# =========================
st.set_page_config(page_title="Teeth Classification", page_icon="🦷", layout="wide")
st.title("🦷 Teeth Classification - Streamlit App")


# =========================
# HELPERS
# =========================
def must_exist(path, msg):
    if not os.path.exists(path):
        st.error(msg)
        st.stop()

def load_classes():
    must_exist(CLASSES_PATH, f"❌ classes.txt not found: {CLASSES_PATH}\n\nشغّل training عشان يتعمل classes.txt")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    if len(names) == 0:
        st.error("❌ classes.txt is empty")
        st.stop()
    return names

class ScratchCNN(nn.Module):
    """بسيطة + قوية كبداية (زي اللي غالبًا عملتها قبل كده)"""
    def __init__(self, num_classes: int):
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


@st.cache_resource
def load_model(model_type: str, model_path: str, num_classes: int):
    must_exist(model_path, f"❌ Model not found: {model_path}\n\nشغّل training الأول عشان يطلع .pth")

    if model_type == "Pretrained (ResNet18)":
        model = models.resnet18(weights=None)  # weights None لأننا هنحمّل وزننا
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = ScratchCNN(num_classes)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    # لو كنت عامل normalization أثناء التدريب قولّي عشان أضيفه هنا
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def predict_single_image(model, image_pil: Image.Image, class_names):
    tfm = get_transform()
    x = tfm(image_pil.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0].detach().cpu().numpy()

    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def plot_probabilities(class_names, probs, topk=7):
    pairs = list(zip(class_names, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:topk]

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig = plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Top Probabilities")
    plt.tight_layout()
    return fig


def run_eval(model, split: str, class_names, batch_size=16):
    split_dir = os.path.join(DATA_ROOT, split)
    must_exist(split_dir, f"❌ Split folder not found: {split_dir}")

    ds = datasets.ImageFolder(root=split_dir, transform=get_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)
            pred = torch.argmax(out, dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float((all_preds == all_labels).mean() * 100.0)
    rep = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm  = confusion_matrix(all_labels, all_preds)
    return acc, rep, cm


def save_eval_outputs(prefix: str, split: str, acc: float, rep: str, cm: np.ndarray, class_names):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 1) report txt
    report_path = os.path.join(OUTPUT_DIR, f"classification_report_{prefix}_{split.lower()}_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {prefix}\nSplit: {split}\nAccuracy: {acc:.2f}%\n\n")
        f.write(rep)

    # 2) metrics json
    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{prefix}_{split.lower()}_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"model": prefix, "split": split, "accuracy": acc}, f, indent=2)

    # 3) confusion matrix png (matplotlib)
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix ({prefix} - {split})")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{prefix}_{split.lower()}_{ts}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    return report_path, metrics_path, cm_path


def list_outputs_files():
    if not os.path.exists(OUTPUT_DIR):
        return []
    files = sorted(os.listdir(OUTPUT_DIR))
    return [os.path.join(OUTPUT_DIR, f) for f in files]


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Settings")

class_names = load_classes()
num_classes = len(class_names)

model_type = st.sidebar.selectbox(
    "Choose Model",
    ["Pretrained (ResNet18)", "Scratch CNN"]
)

# choose model path based on availability
if model_type == "Pretrained (ResNet18)":
    default_path = PRETRAINED_BEST if os.path.exists(PRETRAINED_BEST) else PRETRAINED_LAST
else:
    default_path = SCRATCH_BEST if os.path.exists(SCRATCH_BEST) else SCRATCH_LAST

model_path = st.sidebar.text_input("Model path (.pth)", value=default_path)

eval_split = st.sidebar.selectbox("Evaluation Split", ["Validation", "Testing"])
batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=128, value=16)


# =========================
# LOAD MODEL
# =========================
model = load_model(model_type, model_path, num_classes)


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Evaluate", "📁 Outputs & Reports"])


# -------------------------
# TAB 1: PREDICT
# -------------------------
with tab1:
    st.subheader("Prediction")
    uploaded = st.file_uploader("Upload dental image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict الآن ✅"):
            label, conf, probs = predict_single_image(model, img, class_names)
            st.success(f"Prediction: **{label}** | Confidence: **{conf*100:.2f}%**")

            colA, colB = st.columns([1, 1])
            with colA:
                fig = plot_probabilities(class_names, probs, topk=min(7, len(class_names)))
                st.pyplot(fig)

            with colB:
                st.write("All probabilities:")
                for n, p in sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True):
                    st.write(f"- {n}: {p*100:.2f}%")


# -------------------------
# TAB 2: EVALUATE (AND SAVE)
# -------------------------
with tab2:
    st.subheader(f"Evaluate on {eval_split}")

    st.info("لو الجهاز CPU، الـ Evaluation ممكن ياخد وقت حسب حجم الداتا.")

    if st.button("Run Evaluation + Save to outputs ✅"):
        with st.spinner("Evaluating..."):
            acc, rep, cm = run_eval(model, eval_split, class_names, batch_size=int(batch_size))

        st.success(f"Accuracy: {acc:.2f}%")

        st.text("Classification Report:")
        st.text(rep)

        # show confusion matrix
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix ({model_type} - {eval_split})")
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        plt.yticks(range(len(class_names)), class_names)
        plt.colorbar()
        plt.tight_layout()
        st.pyplot(fig)

        # save
        prefix = "pretrained_resnet18" if model_type == "Pretrained (ResNet18)" else "scratch_cnn"
        report_path, metrics_path, cm_path = save_eval_outputs(prefix, eval_split, acc, rep, cm, class_names)

        st.write("Saved files:")
        st.code(report_path)
        st.code(metrics_path)
        st.code(cm_path)


# -------------------------
# TAB 3: OUTPUTS
# -------------------------
with tab3:
    st.subheader("outputs folder viewer")

    files = list_outputs_files()
    if len(files) == 0:
        st.warning("outputs folder is empty.")
    else:
        # show key images and txt
        imgs = [f for f in files if f.lower().endswith(".png")]
        txts = [f for f in files if f.lower().endswith(".txt")]
        jsons = [f for f in files if f.lower().endswith(".json")]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🖼️ Confusion Matrices / Images")
            if imgs:
                chosen_img = st.selectbox("Choose image", imgs)
                st.image(chosen_img, use_container_width=True)
            else:
                st.info("No PNG images found in outputs.")

        with col2:
            st.markdown("### 📄 Reports (TXT)")
            if txts:
                chosen_txt = st.selectbox("Choose report", txts)
                with open(chosen_txt, "r", encoding="utf-8", errors="ignore") as f:
                    st.text(f.read())
            else:
                st.info("No TXT reports found in outputs.")

        st.markdown("### 🧾 Metrics / Comparison (JSON)")
        if jsons:
            chosen_json = st.selectbox("Choose json", jsons)
            with open(chosen_json, "r", encoding="utf-8", errors="ignore") as f:
                st.json(json.load(f))
        else:
            st.info("No JSON files found in outputs.")

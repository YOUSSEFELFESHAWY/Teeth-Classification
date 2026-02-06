import os
import json

OUTPUT_DIR = "outputs"

PRETRAINED_METRICS = os.path.join(OUTPUT_DIR, "metrics_pretrained_validation.json")
SCRATCH_METRICS    = os.path.join(OUTPUT_DIR, "metrics_scratch_validation.json")

TXT_OUT  = os.path.join(OUTPUT_DIR, "comparison_validation.txt")
JSON_OUT = os.path.join(OUTPUT_DIR, "comparison_validation.json")


def load_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load metrics
pretrained = load_metrics(PRETRAINED_METRICS)
scratch    = load_metrics(SCRATCH_METRICS)

pre_acc = pretrained["accuracy"]
scr_acc = scratch["accuracy"]
diff = pre_acc - scr_acc

# =========================
# PRINT TO TERMINAL
# =========================
print("\n==============================")
print(" Scratch vs Pretrained (Validation)")
print("==============================")
print(f"Pretrained Accuracy : {pre_acc:.2f}%")
print(f"Scratch Accuracy    : {scr_acc:.2f}%")
print(f"Accuracy Difference : {diff:.2f}%")
print("==============================\n")

# =========================
# SAVE TXT
# =========================
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("Scratch vs Pretrained (Validation)\n")
    f.write("=" * 35 + "\n")
    f.write(f"Pretrained Accuracy : {pre_acc:.2f}%\n")
    f.write(f"Scratch Accuracy    : {scr_acc:.2f}%\n")
    f.write(f"Accuracy Difference : {diff:.2f}%\n")

# =========================
# SAVE JSON
# =========================
comparison = {
    "pretrained_accuracy": pre_acc,
    "scratch_accuracy": scr_acc,
    "difference": diff
}

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

print(f"✅ Comparison TXT saved to: {TXT_OUT}")
print(f"✅ Comparison JSON saved to: {JSON_OUT}")

import os
import cv2
import numpy as np
from collections import defaultdict

DATA_ROOT = r"Teeth DataSet\Teeth_Dataset"
SPLITS = ["Training", "Validation", "Testing"]

def iter_images(split_path):
    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        if not os.path.isdir(cls_path):
            continue
        for fn in os.listdir(cls_path):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                yield cls, os.path.join(cls_path, fn)

def main():
    report = defaultdict(list)
    shapes = []

    for split in SPLITS:
        split_path = os.path.join(DATA_ROOT, split)
        if not os.path.exists(split_path):
            print(f"[WARN] Missing split folder: {split_path}")
            continue

        for cls, path in iter_images(split_path):
            img = cv2.imread(path)
            if img is None:
                report["corrupted_images"].append(path)
                continue

            h, w = img.shape[:2]
            c = 1 if img.ndim == 2 else img.shape[2]

            # Missing/invalid checks
            if h < 32 or w < 32:
                report["too_small"].append(path)
            if c not in (3, 4):
                report["weird_channels"].append((path, c))

            shapes.append((h, w))

    print("\n===== DATA CHECKS REPORT =====")
    print("Corrupted images:", len(report["corrupted_images"]))
    print("Too small images:", len(report["too_small"]))
    print("Weird channels:", len(report["weird_channels"]))

    if shapes:
        hs = np.array([s[0] for s in shapes])
        ws = np.array([s[1] for s in shapes])

        # Simple “outlier” detection by IQR (not perfect but useful)
        def iqr_outliers(arr):
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return low, high

        h_low, h_high = iqr_outliers(hs)
        w_low, w_high = iqr_outliers(ws)

        out_h = np.where((hs < h_low) | (hs > h_high))[0]
        out_w = np.where((ws < w_low) | (ws > w_high))[0]
        out_idx = set(out_h.tolist() + out_w.tolist())

        print("\nImage size stats:")
        print(f"Height min/median/max: {hs.min()}/{int(np.median(hs))}/{hs.max()}")
        print(f"Width  min/median/max: {ws.min()}/{int(np.median(ws))}/{ws.max()}")
        print(f"Potential size outliers: {len(out_idx)} (IQR rule)")

    # Print a few examples (optional)
    for k, v in report.items():
        if len(v) > 0:
            print(f"\n--- {k} (showing up to 5) ---")
            for item in v[:5]:
                print(item)

if __name__ == "__main__":
    main()

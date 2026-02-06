import os
import random
import matplotlib.pyplot as plt
import cv2

TRAIN_PATH = r"Teeth DataSet\Teeth_Dataset\Training"

def count_classes(path):
    counts = {}
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
    return counts

def show_samples(path, n_per_class=1):
    classes = [c for c in os.listdir(path) if os.path.isdir(os.path.join(path, c))]
    classes.sort()

    plt.figure(figsize=(3*n_per_class, 3*len(classes)))
    idx = 1

    for cls in classes:
        cls_path = os.path.join(path, cls)
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))]
        if not imgs:
            continue

        chosen = random.sample(imgs, k=min(n_per_class, len(imgs)))
        for fn in chosen:
            img = cv2.imread(os.path.join(cls_path, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(classes), n_per_class, idx)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            idx += 1

    plt.tight_layout()
    plt.show()

def main():
    counts = count_classes(TRAIN_PATH)

    plt.figure(figsize=(10,5))
    plt.bar(counts.keys(), counts.values())
    plt.title("Training Class Distribution")
    plt.xlabel("Class (Illness)")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    show_samples(TRAIN_PATH, n_per_class=1)

if __name__ == "__main__":
    main()

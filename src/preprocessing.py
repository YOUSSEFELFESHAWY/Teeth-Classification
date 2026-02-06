import cv2
import os
import matplotlib.pyplot as plt
import random

DATASET_PATH = r"Teeth DataSet\Teeth_Dataset\Training"

# اختر Class عشوائي
class_name = random.choice(os.listdir(DATASET_PATH))
class_path = os.path.join(DATASET_PATH, class_name)

# اختر صورة عشوائية
img_name = random.choice(os.listdir(class_path))
img_path = os.path.join(class_path, img_name)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Augmentations بسيطة
flip = cv2.flip(img, 1)              # Horizontal flip
rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# عرض الصور
plt.figure(figsize=(9, 3))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(flip)
plt.title("Flipped")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(rot)
plt.title("Rotated")
plt.axis("off")

plt.suptitle(f"Class: {class_name}")
plt.tight_layout()
plt.show()

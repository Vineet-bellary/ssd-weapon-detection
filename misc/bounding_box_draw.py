import json
import cv2
import matplotlib.pyplot as plt

ANN_PATH = "CGI-Weapon-Dataset-1/valid/_annotations.coco.json"
IMG_DIR = "CGI-Weapon-Dataset-1/valid"

# Load COCO annotations
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

# Pick first image
image_info = coco["images"][0]
image_id = image_info["id"]
file_name = image_info["file_name"]

img_path = f"{IMG_DIR}/{file_name}"
print("Image path:", img_path)

# Load image
image = cv2.imread(img_path)
if image is None:
    raise ValueError("Image not loaded")

# Convert BGR → RGB (IMPORTANT)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Collect annotations for this image
image_anns = []
for ann in coco["annotations"]:
    if ann["image_id"] == image_id:
        image_anns.append(ann)

print(f"Total objects: {len(image_anns)}")

# step 4
ssd_labels = []  # <-- NEW (labels list)

# Draw bounding boxes (but IGNORE Weapons = 0)
for ann in image_anns:
    if ann["category_id"] == 0:
        continue  # <-- Weapons ni skip cheyyi

    ssd_labels.append(ann["category_id"])  # <-- NEW (store label)

    x, y, w, h = ann["bbox"]

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

print("Final SSD labels:", ssd_labels)

# SHOW IMAGE
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.show()

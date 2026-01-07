import json

# ==============================
# PATHS
# ==============================
ANN_PATH = "CGI-Weapon-Dataset-1/valid/_annotations.coco.json"

# ==============================
# LOAD COCO JSON
# ==============================
with open(ANN_PATH, "r") as f:
    coco = json.load(f)

# ==============================
# PICK ONE IMAGE (for experiment)
# ==============================
image_info = coco["images"][0]

image_id = image_info["id"]
file_name = image_info["file_name"]
img_w = image_info["width"]
img_h = image_info["height"]

print("IMAGE INFO")
print(image_info)
print("-" * 50)

# ==============================
# COLLECT ALL ANNOTATIONS
# FOR THIS IMAGE
# ==============================
image_anns = []

for ann in coco["annotations"]:
    if ann["image_id"] == image_id:
        image_anns.append(ann)

print(f"Total objects in image: {len(image_anns)}")
print("-" * 50)

# ==============================
# COCO -> SSD CONVERSION
# ==============================
ssd_boxes = []
ssd_labels = []

for ann in image_anns:
    x, y, w, h = ann["bbox"]

    # center format
    cx = x + w / 2
    cy = y + h / 2

    # normalize
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm  = w / img_w
    h_norm  = h / img_h

    ssd_boxes.append((cx_norm, cy_norm, w_norm, h_norm))
    ssd_labels.append(ann["category_id"])

# ==============================
# OUTPUT
# ==============================
print("SSD BOXES (cx, cy, w, h):")
for box in ssd_boxes:
    print(box)

print("\nSSD LABELS:")
print(ssd_labels)

print("\nDone ✅")

import torch
from anchors import generate_anchors, compute_iou
from dataset import WeaponSSDDataset

# ============================
# LOAD DATASET (GT BOXES)
# ============================
dataset = WeaponSSDDataset(
    ann_path="CGI-Weapon-Dataset-1/valid/_annotations.coco.json",
    img_dir="CGI-Weapon-Dataset-1/valid"
)

# Take ONE image sample
_, gt_boxes, labels = dataset[0]   # gt_boxes: [N, 4], labels: [N]

# ============================
# GENERATE ANCHORS
# ============================
anchors = generate_anchors()        # [A, 4]

# ============================
# IOU MATRIX
# ============================
iou_matrix = compute_iou(anchors, gt_boxes)   # [A, N]

print("IoU matrix shape:", iou_matrix.shape)

# ============================
# STEP 9.1: BEST GT FOR EACH ANCHOR
# ============================
best_iou_per_anchor, best_gt_idx = iou_matrix.max(dim=1)

# ============================
# STEP 9.2: POSITIVE / NEGATIVE MASKS
# ============================
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = 0.4

positive_mask = best_iou_per_anchor >= POSITIVE_THRESHOLD
negative_mask = best_iou_per_anchor < NEGATIVE_THRESHOLD

# ============================
# STEP 9.3: FORCE ONE POSITIVE PER GT
# ============================
best_anchor_per_gt = iou_matrix.argmax(dim=0)

positive_mask[best_anchor_per_gt] = True
negative_mask[best_anchor_per_gt] = False

print("Positive anchors:", positive_mask.sum().item())
print("Negative anchors:", negative_mask.sum().item())

# ============================
# STEP 9.4: CLASSIFICATION TARGETS
# ============================
num_anchors = anchors.size(0)

# background = 0
cls_targets = torch.zeros(num_anchors, dtype=torch.long)

# assign GT labels to positive anchors
cls_targets[positive_mask] = labels[best_gt_idx[positive_mask]]

# ============================
# STEP 9.5: LOCALIZATION TARGETS
# ============================
matched_gt_boxes = gt_boxes[best_gt_idx]

tx = (matched_gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
ty = (matched_gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
tw = torch.log(matched_gt_boxes[:, 2] / anchors[:, 2])
th = torch.log(matched_gt_boxes[:, 3] / anchors[:, 3])

loc_targets = torch.stack([tx, ty, tw, th], dim=1)

print("Localization targets shape:", loc_targets.shape)

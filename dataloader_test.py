import torch
from torch.utils.data import DataLoader
from dataset import WeaponSSDDataset


def ssd_collate_fn(batch):
    images = []
    boxes = []
    labels = []

    for img, bxs, lbls in batch:
        images.append(img)
        boxes.append(bxs)
        labels.append(lbls)

    images = torch.stack(images, dim=0)
    return images, boxes, labels


# ------------------------------------------------------------

dataset = WeaponSSDDataset(
    ann_path="CGI-Weapon-Dataset-1/valid/_annotations.coco.json",
    img_dir="CGI-Weapon-Dataset-1/valid",
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=ssd_collate_fn)

# ------------------------------------------------------------
images, boxes, labels = next(iter(loader))

print(images.shape)  # [2, 3, 640, 640]
print(len(boxes))  # 2
print(boxes[0].shape)  # [N, 4]
print(labels[0])

import torch
import torch.nn as nn
import torchvision.models as models


class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # -------------------------
        # BACKBONE: VGG16
        # -------------------------
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = vgg.features  # conv layers only

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # -------------------------
        # SSD HEAD
        # -------------------------
        self.num_classes = num_classes
        self.num_anchors = 4  # must match anchors.py

        self.cls_head = nn.Conv2d(
            512, self.num_anchors * num_classes, kernel_size=3, padding=1
        )

        self.loc_head = nn.Conv2d(
            512, self.num_anchors * 4, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, 512, 20, 20]

        cls = self.cls_head(x)
        loc = self.loc_head(x)

        B = x.size(0)

        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(B, -1, self.num_classes)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(B, -1, 4)

        return cls, loc


if __name__ == "__main__":
    model = SSD(num_classes=4)
    dummy = torch.randn(2, 3, 640, 640)
    cls_preds, loc_preds = model(dummy)
    print(cls_preds.shape)  # [2, 1600, 4]
    print(loc_preds.shape)  # [2, 1600, 4]

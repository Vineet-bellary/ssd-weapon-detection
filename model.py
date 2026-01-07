import torch
import torch.nn as nn


class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 640 -> 320
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 320 -> 160
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 160 -> 80
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 80 -> 40
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)  # [B, 256, 20, 20]


class SSDHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = 16

        self.cls_head = nn.Conv2d(
            256, self.num_anchors * num_classes, kernel_size=3, padding=1
        )

        self.loc_head = nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        B = x.size(0)

        cls = self.cls_head(x)
        loc = self.loc_head(x)

        # reshape
        cls = cls.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(B, -1, self.num_classes)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(B, -1, 4)

        return cls, loc


class SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = SimpleBackbone()
        self.head = SSDHead(num_classes)

    def forward(self, x):
        features = self.backbone(x)
        cls_preds, loc_preds = self.head(features)
        return cls_preds, loc_preds


if __name__ == "__main__":
    model = SSD(num_classes=4)

    dummy = torch.randn(2, 3, 640, 640)
    cls_preds, loc_preds = model(dummy)

    print(cls_preds.shape)  # [2, 1600, 4]
    print(loc_preds.shape)  # [2, 1600, 4]

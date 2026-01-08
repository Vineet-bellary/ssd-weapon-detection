# SSD Weapon Detection (From Scratch)

This project implements a **Single Shot Detector (SSD)** object detection model from scratch using **PyTorch**.

The model is trained to detect the following weapon classes:

- pistol
- rifle
- shotgun

This project is created mainly for **learning purposes** to understand how SSD works internally, including anchors, IoU matching, loss computation, training, inference, and evaluation.

---

## Project Structure

SSD_tutorial/
│
├── anchors.py # Anchor generation logic
├── model.py # SSD model architecture
├── dataset.py # Dataset loader (COCO format)
├── ssd_loss.py # SSD loss (classification + localization)
│
├── train.py # Training script
├── inference.py # Inference & visualization
├── evaluate.py # Precision, Recall, F1-score evaluation
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
└── CGI-Weapon-Dataset-1/
├── train/
│ ├── images/
│ └── \_annotations.coco.json
│
├── valid/
│ ├── images/
│ └──_annotations.coco.json
│
└── test/
└── images/

---

## Dataset

- Dataset annotations are in **COCO format**
- Classes used:
  - `1 → pistol`
  - `2 → rifle`
  - `3 → shotgun`
- Background class is handled internally as class `0`

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Optional but recommended

```bash
python -m venv  .venv
.venv\Scripts\activate

```

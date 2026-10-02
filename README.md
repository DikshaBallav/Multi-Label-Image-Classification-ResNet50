# 🖼️ Multi-Label Image Classification with ResNet-50

> A deep learning project for multi-label image classification using ResNet-50 transfer learning with ImageNet pretrained weights.

![Python](https://img.shields.io/badge/Python-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-green)
![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-purple)

---

## 📌 Project Overview

Developed a multi-label image classification model using **ResNet-50 transfer learning** with ImageNet pretrained weights.

Handled missing labels using **mask-aware loss** and addressed class imbalance using **weighted Binary Cross Entropy**.

> Developed as part of an AI/ML assignment for **Aimonk**.

---

## 🎯 Problem Statement

This project solves a multi-label image classification task where:

- Each image may contain multiple attributes simultaneously.
- Labels are provided in `labels.txt`.
- Some attribute values are missing (`NA`) and must **NOT** be ignored.
- Dataset is imbalanced, requiring mitigation during training.
- Model must be fine-tuned from ImageNet pretrained weights (not trained from scratch).

---

## 🗂️ Dataset Structure

```text
project/
│
├── images/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
│
├── labels.txt
└── notebook.ipynb
```

### Label Format

```text
Image_Name   Attr1   Attr2   Attr3   Attr4

image_0.jpg  1       NA      0       1
```

### Value Meaning

| Value | Meaning |
|---|---|
| `1` | Attribute present |
| `0` | Attribute absent |
| `NA` | Unknown label (must be handled, not dropped) |

---

## 🧠 Approach

### Model

1. **ResNet-50 (ImageNet pretrained)** used for transfer learning.
2. Final FC layer replaced with **4-node sigmoid output**.

### Model Flow

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
ResNet-50
(ImageNet Pretrained)
     │
     ▼
Modified Final FC Layer
     │
     ▼
4-Node Sigmoid Output
     │
     ▼
Multiple Attribute Predictions
```

---

## 💡 Why Transfer Learning?

Training from scratch would:

1. Require large dataset
2. Overfit quickly
3. Be computationally expensive

Fine-tuning leverages learned visual features.

---

## 🔍 Handling Missing Labels (NA)

Instead of removing samples:

1. `NA` values are masked during loss computation.
2. Loss is calculated only where labels are known.

This allows:

1. Use of full dataset
2. No bias from dropping images

### Masking Logic

```text
Label = 1  → Included in loss
Label = 0  → Included in loss
Label = NA → Masked from loss
```

---

## ⚖️ Handling Class Imbalance

Dataset is skewed, so we use:

### Weighted Binary Cross Entropy

- `pos_weight` computed per attribute

This prevents the model from always predicting the majority class.

---

## 🧹 Data Preprocessing

### Image Processing

1. Resize → `224 × 224`
2. Normalize using ImageNet statistics

### Augmentations

- Random Horizontal Flip
- Mild Rotation
- Color jitter (helps generalization)

---

## 📊 Training Output

A required plot is generated:

```text
Title: Aimonk_multilabel_problem

X-axis: iteration_number
Y-axis: training_loss
```

### Output

```text
Attributes Present: ['Attr1', 'Attr4']
```

---

## 🔬 Techniques Considered (Time-Constrained)

The following can further improve results:

- Focal Loss for extreme imbalance
- Test-time augmentation
- Label smoothing
- Attribute-wise threshold tuning
- Using EfficientNet instead of ResNet
- Semi-supervised learning to exploit `NA` labels

---

## 🧩 Code Design Philosophy

- Modular dataset loader
- Mask-aware loss computation
- Easily swappable backbone

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Architecture | ResNet-50 |
| Learning Approach | Transfer Learning |
| Domain | Computer Vision |
| Classification | Multi-Label Classification |
| Loss Function | Weighted Binary Cross Entropy |

---

## 📁 Repository Contents

```text
Multi-Label-Image-Classification-ResNet50/
│
├── README.md
└── train.ipynb
```

---

## 🚀 Key Highlights

### 🧠 Deep Learning

ResNet-50 with ImageNet pretrained weights for transfer learning.

### 🔍 Missing Label Handling

Mask-aware loss computation allows training with unknown (`NA`) labels without dropping samples.

### ⚖️ Class Imbalance

Weighted Binary Cross Entropy with attribute-specific `pos_weight`.

### 🖼️ Multi-Label Prediction

The model predicts multiple attributes for a single image using a 4-node sigmoid output.

---

## 👩‍💻 Author

### Diksha Ballav

**AI/ML | Machine Learning | Deep Learning | NLP**

# Multilabel-Image-Classification-Aimonk-Assignment-
# Problem Statement-

This project solves a multi-label image classification task where:

Each image may contain multiple attributes simultaneously.

Labels are provided in labels.txt.

Some attribute values are missing (NA) and must NOT be ignored.

Dataset is imbalanced, requiring mitigation during training.

Model must be fine-tuned from ImageNet pretrained weights (not trained from scratch).

# Dataset Structure- 
project/
|-- images/
│   ├── image_0.jpg
│   ├── image_1.jpg
│   └── ...
│
|-- labels.txt
|-- notebook.ipynb

Label Format
Image_Name
Attr1
Attr2
Attr3
Attr4

image_0.jpg
1
NA
0
1
Value	Meaning
1	Attribute present
0	Attribute absent
NA	Unknown label (must be handled, not dropped)

# Approach-
# Model

1) ResNet-50 (ImageNet pretrained) used for transfer learning.

2) Final FC layer replaced with 4-node sigmoid output.

# Why Transfer Learning?

Training from scratch would:
1) Require large dataset

2) Overfit quickly

3) Be computationally expensive

Fine-tuning leverages learned visual features.

# Handling Missing Labels (NA)

Instead of removing samples:

1)NA values are masked during loss computation.

2)Loss is calculated only where labels are known.

This allows:
1) Use of full dataset
2)No bias from dropping images

# Handling Class Imbalance-

Dataset is skewed, so we use:

   Weighted Binary Cross Entropy

   pos_weight computed per attribute

This prevents model from always predicting majority class.

# Data Preprocessing-

1)Resize → 224×224

2)Normalize using ImageNet statistics

3)Augmentations:

  Random Horizontal Flip

  Mild Rotation

  Color jitter (helps generalization)

# Training Output-

A required plot is generated:

   Title: Aimonk_multilabel_problem

   X-axis: iteration_number

   Y-axis: training_loss

# Output:

Attributes Present: ['Attr1', 'Attr4']
# Techniques Considered (Time-Constrained)

The following can further improve results:

   Focal Loss for extreme imbalance

   Test-time augmentation
   
   Label smoothing

   Attribute-wise threshold tuning

   Using EfficientNet instead of ResNet

   Semi-supervised learning to exploit NA labels

# Code Design Philosophy

  Modular dataset loader

  Mask-aware loss computation

  Easily swappable backbone





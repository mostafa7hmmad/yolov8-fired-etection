# mostafa7hmmad-yolov8-fired-etection
# 🔥 Fire and Smoke Detection with YOLOv8

Fire and smoke detection is critical for early warning systems in wildfire monitoring and safety applications. This project implements a deep learning solution using the *YOLOv8s* model from Ultralytics for accurate and real-time detection of fire and smoke in images. The model is trained on a *custom dataset* and evaluated using various performance metrics to ensure real-world applicability.

---

## 🛠 Preprocessing Steps

* *Image Scanning*: Training and validation datasets are scanned for images and labels.
* *Corrupt Image Handling*: 21 corrupt training images and 5 corrupt validation images are ignored.
* *Normalization: All images are resized to **640x640* pixels for YOLOv8 compatibility.
* *Dataset Configuration*: A data.yaml file defines image paths and class labels (fire, smoke).

---

## 🔀 Data Splitting

* *Training Set*: 14,101 images (after removing corrupt files)
* *Validation Set*: 3,094 images (after removing corrupt files)
* *No Test Split*: Model evaluation is done on the validation set.

---

## 🔄 Data Augmentation

To enhance model generalization and robustness:

### Training Set:

* Mosaic (80% probability)
* Random Rotation (±10°)
* Horizontal Flip (50%)
* HSV Adjustments:

  * Hue: 0.015
  * Saturation: 0.7
  * Value: 0.4
* Random Scale: 0.7
* Shear: 0.3
* Translate: 0.2
* *Albumentations*:

  * Blur
  * Median Blur
  * ToGray
  * CLAHE

### Validation Set:

* No augmentation, only normalization.

---

## 🧠 Model Details

* *Architecture*: YOLOv8s (small variant)

* *Parameters*: 11.14M

* *Layers*: 129

* *GFLOPs*: 28.6

* *Classes*: fire, smoke

* *Loss Functions*:

  * *Box Loss*: Bounding box regression
  * *Classification Loss*: Class prediction
  * *DFL (Distribution Focal Loss)*: Localization improvement

* *Evaluation Metrics*:

  * Precision
  * Recall
  * mAP\@0.5
  * mAP\@0.5:0.95

---

## 🏋 Training Strategy

* *Optimizer*: Adam
* *Learning Rate*: 0.0005
* *Momentum*: 0.937
* *Batch Size*: 32
* *Epochs*: 30
* *Early Stopping*: After 7 epochs without improvement
* *Hardware*: Tesla P100 GPU with CUDA 12.4
* *Fine-Tuning*: Mosaic disabled in the last 10 epochs

---

## 📊 Evaluation

Evaluation is performed using the validation set. Metrics:

* ✅ Precision
* ✅ Recall
* ✅ mAP\@0.5
* ✅ mAP\@0.5:0.95
* ✅ Box, Classification, and DFL losses

### 📌 Sample Results (Epoch 10):

* *Precision*: 0.663
* *Recall*: 0.604
* *mAP\@0.5*: 0.660
* *mAP\@0.5:0.95*: 0.334

---

## 📈 Visualization of Training Metrics

To assess model performance visually:

* Load results.csv
* Plot training and validation losses:

  * Box Loss
  * Classification Loss
  * DFL Loss
* Display all plots side-by-side using Matplotlib to compare trends over epochs

---

## 🔍 Pipeline Diagram

mermaid
flowchart TD
  A[Start: Fire and Smoke Detection] --> B[Data Collection]
  B --> B1[Dataset: Smoke-Fire-Detection-YOLO]
  B --> B2[Train: 14,101 Images, Val: 3,094 Images]
  B --> B3[Classes: Fire, Smoke]

  B3 --> C[Preprocessing]
  C --> C1[Scan Images and Labels]
  C --> C2[Ignore 21 Train, 5 Val Corrupt Images]
  C --> C3[Resize to 640x640]

  C3 --> D[Splitting]
  D --> D1[Train: 14,101 Images]
  D --> D2[Val: 3,094 Images]

  D2 --> E[Augmentation]
  E --> E1[Train: Mosaic, Flip, Rotate, HSV, Albumentations]
  E --> E2[Val: Normalize Only]

  E2 --> F[Model]
  F --> F1[YOLOv8s, 11.14M Params]
  F --> F2[Loss: Box, Cls, DFL]
  F --> F3[Metrics: Precision, Recall, mAP]

  F3 --> G[Training]
  G --> G1[Adam Optimizer, lr=0.0005]
  G --> G2[Batch Size: 32]
  G --> G3[Early Stop after 7 Epochs No Improvement]
  G --> G4[Max Epochs: 30]

  G4 --> H[Evaluation]
  H --> H1[Precision, Recall]
  H --> H2[mAP50, mAP50-95]
  H --> H3[Loss Metrics]

  H3 --> I[Visualization]
  I --> I1[Load results.csv]
  I --> I2[Plot Box, Cls, DFL Losses]
  I --> I3[Display Train vs. Val Trends]


---

## 📦 Requirements

* Python 3.11
* PyTorch 2.6.0 (with CUDA 12.4)
* Ultralytics 8.3.135
* Pandas
* Matplotlib
* Dataset: *Smoke and Fire Detection YOLO*

---

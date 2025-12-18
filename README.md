# 🫁 Tuberculosis Lesion Segmentation using YOLOv8
### A Data Mining Approach to Medical Image Analysis

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=github)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8_Seg-orange?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-red?style=for-the-badge)

## 📖 Introduction & Problem Definition

Tuberculosis (TB) remains one of the top infectious killers worldwide. While Chest X-Rays (CXR) are the standard screening method, manual interpretation is subjective, time-consuming, and prone to human error.

Most automated solutions focus on **Binary Classification** (TB vs. Healthy). However, for clinical support, it is crucial not just to detect the disease but to **localize** it.

**This project aims to:**
1.  Implement an instance segmentation model using the **YOLOv8** architecture.
2.  Segment and visualize TB lesions directly on X-ray images.
3.  Demonstrate the Data Mining pipeline: from Data Collection → Preprocessing → Modeling → Inference.

---

## 🤖 Model Architecture: Why YOLOv8?

For this project, we selected **YOLOv8 (You Only Look Once - Version 8)**, specifically the Segmentation variant (`yolov8-seg`).

### Key Features:
* **Single-Shot Detection:** Unlike older R-CNN models that process images in multiple steps, YOLO looks at the entire image once, making it extremely fast.
* **Anchor-Free Detection:** It predicts the center of an object directly, reducing the number of hyperparameters and improving accuracy on irregular shapes (like lung lesions).
* **Segmentation Head:** Beyond drawing a simple box, YOLOv8-seg outputs a **binary mask** that outlines the exact contour of the infection.

**Why not simple Classification?**
A classification model only answers *"Is there TB?"*. Our YOLO segmentation model answers *"Where is the TB and how large is the affected area?"*, providing much higher clinical value.

---

## 🧠 Training Strategy (Methodology)

We utilized **Transfer Learning** to overcome the limitation of a relatively small medical dataset. Instead of training from scratch, we started with weights pre-trained on the COCO dataset (common objects) and fine-tuned them for medical X-rays.

### ⚙️ Hyperparameters & Configuration
* **Base Model:** `yolov8n-seg.pt` (Nano version) - chosen for efficiency and low memory usage.
* **Image Resolution:** **512x512 pixels** (Resized from raw resolution to balance detail vs. speed).
* **Epochs:** **50** (Sufficient for convergence without overfitting).
* **Batch Size:** 16.
* **Optimizer:** SGD (Stochastic Gradient Descent) with momentum.

### The Learning Process
The model optimizes two loss functions simultaneously:
1.  **Box Loss:** How accurately it locates the lesion area.
2.  **Seg Loss (Mask Loss):** How accurately the predicted polygon matches the actual shape of the lesion.

---

## 📂 Project Deliverables Roadmap

This repository represents the continuous assessment work for the **Data Mining** course:

### ✅ Deliverable 1: Data Acquisition
* **Data Sources:** Integration of **Shenzhen** and **Montgomery** Chest X-Ray datasets.
* **Data Structure:** CXR images + Binary Masks + Clinical Metadata.

### ✅ Deliverable 2: Preprocessing & Transformation
* **Filtering:** Selected only confirmed TB cases using `MetaData.csv`.
* **Mask-to-Polygon Conversion:** Custom OpenCV pipeline to transform binary masks into YOLO `.txt` polygon format.
* **Data Split:** 80% Training / 20% Validation.

### ✅ Final Deliverable: Results
* **Inference:** The model generates masks that closely align with radiologist annotations.
* **Visual Validation:** Overlays of predicted masks on test images demonstrate successful localization.

---


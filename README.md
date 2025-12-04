# Lightweight_Drone_Models
In-progress project comparing lightweight drone models for detecting humans, animals, and hazards in forest fires and rescue operations.

# Overview

This project implements a lightweight object detection model based on SSDLite MobileNet V3 for drones used in forest fire monitoring and rescue missions. The system is designed to identify objects of interest (such as humans, animals, and fire hazards) in aerial imagery while maintaining a low computational footprint suitable for deployment on edge devices like drones.

# Dataset

The model is trained on the COCO 2017 dataset, which provides images and annotations for various object categories. Key dataset handling steps include:

- Download and extraction of train (14GB) and validation (1GB) images.
- Annotation conversion: Original COCO annotations are converted to YOLO format for compatibility with the MobileNet SSD pipeline.
- Train-validation split: Adjusted to 80:20 for better generalization.
- Image preprocessing: Resized to 320×320 pixels and normalized for MobileNet input.

Dataset statistics:
   - Split	Images
     a) Training	~80% of COCO train2017
     b) Validation	~20% of COCO train2017 + original val2017
# Model

The model used is SSDLite320 MobileNet V3 Large, a lightweight SSD variant optimized for mobile and embedded devices. It includes:
   - Pretrained COCO weights.
   - Training on custom dataset splits.
   - Input resolution of 320×320 pixels.

# Features

1. Fast inference suitable for drone deployment.
2. Compatible with edge devices with limited compute resources.
3. Handles multi-object detection using bounding boxes.

# Training

1. Key training details:
   - Optimizer: Adam with learning rate 0.001.
   - Batch size: 16 images.
   - Epochs: 10.
   - Device: T4 GPU on google collab
   - Loss calculation: Sum of classification and localization losses provided by PyTorch SSD implementation.

2. The training loop monitors:
   - Total loss per epoch.
   - Precision, recall, and mean Average Precision (mAP).
   - Confusion matrix of predictions.
3. Evaluation Metrics
   - Loss vs Epochs: Tracks model convergence.
   - Precision-Recall Curve: Evaluates detection performance across thresholds.
   - mAP (Mean Average Precision): Key metric for object detection.
   - Confusion Matrix: Assesses true positives, false positives, and false negatives.

# Usage

Prepare the dataset in the structure:

coco2017/
├── train2017/
├── val2017/
└── labels/
    ├── train2017/
    └── val2017/


- Run the notebook to train the model or load pretrained weights for inference.
- Use the calculate_metrics function to evaluate the model on validation images.

# Results

1. Calculated mAP on validation set.
2. Precision-Recall curve and confusion matrix plotted for analysis.
3. Loss converges steadily over epochs.

# Future Work
- Implement YOLOv8 and EfficientDet models for comparison.
- Deploy trained models on real drone hardware for forest fire detection.
- Fine-tune models for small object detection and low-light conditions.

# References
 - COCO Dataset: http://cocodataset.org
 - MobileNetV3 SSD: Howard et al., Searching for MobileNetV3, CVPR 2019
 - PyTorch Object Detection: https://pytorch.org/vision/stable/models.html

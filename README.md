# AI-Driven-Pneumonia-Detection-from-Chest-Medical-Images
Automatic Pneumonia Detection in Medical Images Using Deep Neural Networks
AI-Driven Pneumonia Detection from Chest X-Rays
# Overview
This repository presents an AI-based solution for automated pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs). Pneumonia is a major global health concern, and timely diagnosis plays a crucial role in patient outcomes. The proposed system acts as a computer-aided diagnostic (CAD) tool to support healthcare professionals by improving diagnostic accuracy, speed, and consistency.
# Key Features
Deep CNN Architecture: A custom-designed 5-layer convolutional neural network tailored for medical image analysis
Data Augmentation Strategy: Advanced augmentation techniques to address class imbalance and reduce overfitting
Robust Evaluation: Performance assessed using precision, recall, F1-score, and AUC-ROC metrics
Clinical-Oriented Design: Developed with real-world medical constraints and requirements in mind
Production-Ready Pipeline: Modular, scalable, and well-structured codebase
# Model Performance
Overall Test Accuracy: 79.33%
Pneumonia Detection (Class 0)
Precision: 0.7564
Recall: 0.9872
F1-Score: 0.8565
Normal Detection (Class 1)
Precision: 0.9565
Recall: 0.4701
F1-Score: 0.6304
# Project Structure
AI-Driven-Pneumonia-Detection/
├── config.py              # Hyperparameters and configuration paths
├── data_loader.py         # Dataset loading utilities
├── data_preprocessor.py   # Preprocessing and data augmentation
├── model_builder.py       # CNN model architecture
├── train.py               # Training workflow
├── evaluate.py            # Evaluation and metrics computation
├── utils.py               # Utility functions
├── main.py                # End-to-end pipeline execution
└── requirements.txt       # Project dependencies
# Technical Implementation
Neural Network Architecture
Input: 150×150 grayscale chest X-ray images
5 Convolutional Layers with Batch Normalization and Dropout
3 MaxPooling Layers for spatial dimensionality reduction
2 Fully Connected Layers (128 neurons + sigmoid output layer)
Regularization: L2 regularization and aggressive dropout to mitigate overfitting
Data Processing Pipeline
Image preprocessing: grayscale conversion, resizing, and normalization
Class balancing through targeted data augmentation
Augmentation techniques: rotation, shifting, shearing, zooming, and flipping
Stratified train/validation/test split to ensure fair evaluation
# Results Interpretation
The model achieves excellent sensitivity for pneumonia detection, as reflected by a high recall score (0.9872), making it particularly effective for identifying positive cases. However, the lower recall for normal cases indicates a tendency to misclassify some healthy samples as pneumonia. Despite this, the high precision for normal cases (0.9565) suggests a low false-positive rate, highlighting the model’s cautious diagnostic behavior.
# Getting Started
Prerequisites
Python 3.8+
TensorFlow 2.10+
OpenCV
Scikit-learn
Installation
Clone the repository:
git clone https://github.com/yourusername/AI-Driven-Pneumonia-Detection.git

cd AI-Driven-Pneumonia-Detection
Install dependencies:
pip install -r requirements.txt
Prepare the dataset:
chest_xray/
├── train/
│   ├── PNEUMONIA/
│   └── NORMAL/
└── test/
    ├── PNEUMONIA/
    └── NORMAL/
Run the full pipeline:
python main.py
# Advanced Features
Transfer Learning support for pre-trained models (ResNet, DenseNet, etc.)
TensorBoard integration for real-time monitoring
Model checkpointing for best-performing weights
Class weight balancing for imbalanced datasets
Extended evaluation: ROC curves, confusion matrices, precision-recall curves
# Dataset
This project uses the Chest X-Ray Images (Pneumonia) dataset, which includes 5,856 validated images, distributed as follows:
Pneumonia: 3,883 images
Normal: 1,349 images
The data originates from pediatric patients (ages 1–5) at Guangzhou Women and Children’s Medical Center.
# Future Improvements
Integration of transfer learning using state-of-the-art CNN architectures
Multi-class classification for different pneumonia subtypes
Extension to 3D CNNs for CT scan analysis
Real-time deployment via a web-based interface
Explainable AI (XAI) techniques for improved clinical interpretability

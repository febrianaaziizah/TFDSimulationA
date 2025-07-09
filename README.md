# TFDSimulationA
# TFD Simulation & Machine Learning Solutions

This repository contains solutions for five different machine learning problems using TensorFlow and Keras. The problems include basic regression, image classification, transfer learning, sentiment analysis, and time series forecasting.

---

## Problem List & Descriptions

### 🔹 Problem A1: Linear Regression
- Task: Train a simple neural network to learn a linear relationship between X and Y values.
- File: `Problem_A1.py`
- Output: `model_A1.h5`

### 🔹 Problem A2: Horse or Human Classification (CNN)
- Task: Build a Convolutional Neural Network to classify images as either horse or human.
- File: `Problem_A2.py`
- Output: `model_A2.h5`
- Target Accuracy: > 83%

### 🔹 Problem A3: Horse or Human Classification with Transfer Learning (InceptionV3)
- Task: Use pre-trained InceptionV3 for better accuracy on the same horse or human dataset.
- File: `Problem_A3.py`
- Output: `model_A3.h5`
- Target Accuracy: > 97%

### 🔹 Problem A4: IMDB Sentiment Analysis
- Task: Classify movie reviews as positive or negative using text data.
- File: `Problem_A4.py`
- Output: `model_A4.h5`
- Target Accuracy: > 83%

### 🔹 Problem A5: Sunspots Time Series Forecasting
- Task: Predict sunspot activity using Conv1D and LSTM layers.
- File: `Problem_A5.py`
- Output: `model_A5.h5`
- Target MAE: < 0.15

---

## How to Run

1. Install the required libraries:
   ```bash
   pip install tensorflow tensorflow-datasets numpy keras

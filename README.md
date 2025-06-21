# Chest X-Ray Pneumonia Classification

This repository contains a Jupyter Notebook that builds a deep learning model using TensorFlow to classify chest X-ray images as either **Pneumonia** or **Normal**. The dataset is sourced from Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Project Overview

Pneumonia is a common lung infection that can be life-threatening if not diagnosed early. This project leverages convolutional neural networks (CNNs) to automate the detection of pneumonia in chest X-rays.

## Dataset

The dataset used is available at Kaggle:
- **Link**: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
- **Categories**: 
  - `PNEUMONIA`
  - `NORMAL`

The dataset is structured into three folders: `train`, `val`, and `test`.

## Features

- Data loading via KaggleHub.
- Data preprocessing and augmentation using OpenCV and Pillow.
- Model built using TensorFlow and Keras.
- Training with early stopping and validation monitoring.
- Evaluation using accuracy, confusion matrix, and loss curves.

## Installation

Install required libraries:

```bash
pip install numpy pandas matplotlib seaborn opencv-python Pillow scikit-learn tensorflow kagglehub

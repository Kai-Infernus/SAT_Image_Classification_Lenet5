# SAT_Image_Classification_Lenet5

# EuroSAT Land Cover Classification with LeNet-5

This project uses the [EuroSAT dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) for land cover classification via satellite images. The EuroSAT dataset includes RGB images in 10 classes, representing various types of land use and cover, such as forest, agricultural, and residential areas. A modified LeNet-5 architecture is implemented for training, taking advantage of GPU parallelization for faster processing.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview

This project is a classification task on the EuroSAT dataset using a modified LeNet-5 convolutional neural network model. The goal is to accurately classify satellite images into 10 different land cover types. We leverage GPU acceleration and PyTorch's DataParallel for faster training on multiple GPUs.

## Dataset

The EuroSAT dataset includes 10 land cover classes:
- Residential
- Industrial
- River
- Sea/Lake
- Herbaceous Vegetation
- Highway
- Pasture
- Forest
- Annual Crop
- Permanent Crop

### Sample Images

Below are some examples of the images in the EuroSAT dataset:

![image](https://github.com/user-attachments/assets/08c1754f-e755-4e73-ae02-87e6790afd76)

## Model Architecture

The modified LeNet-5 model includes convolutional, pooling, and fully connected layers, adapted to handle RGB satellite images. The architecture is particularly suited for smaller datasets and provides a good balance of accuracy and computational efficiency.

## Results
Training performance is visualized below. The model achieves good accuracy on the EuroSAT dataset, demonstrating effective land cover classification.

# Accuracy vs. Epoch
![image](https://github.com/user-attachments/assets/bb62ba94-4b85-404b-8efd-7dc0ab789d6b)

# Loss vs. Epoch
![image](https://github.com/user-attachments/assets/33bdb337-cd98-48ae-8139-19d3260c6ce4)

## Installation

To run this project, you’ll need Python 3 and the following packages:
- PyTorch
- torchvision
- numpy
- matplotlib

Install the dependencies via pip:

```bash
pip install torch torchvision numpy matplotlib



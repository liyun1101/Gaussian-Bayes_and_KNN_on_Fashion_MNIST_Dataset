# Gaussian-Bayes and KNN on Fashion MNIST Dataset
[![Build Status](https://travis-ci.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset.svg?branch=master)](https://travis-ci.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)
[![GitHub](https://img.shields.io/github/license/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset/blob/master/LICENSE)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset?include_prereleases)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset/releases)
[![GitHub repo size](https://img.shields.io/github/repo-size/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bff26b8d5c544e84a70ca430d1129d57)](https://www.codacy.com/manual/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset&amp;utm_campaign=Badge_Grade)
[![GitHub top language](https://img.shields.io/github/languages/top/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://www.python.org/)

## Overview
This project employed two machine learning methods to classify the fashion MNIST dataset:
 
- ML estimation with Gaussian assumption followed by Bayes rule
- K-Nearest-Neighbor  

Two dimensionality reduction techniques are applied on both machine learning methods: 
 
- PCA (Principal Component Analysis)
- LDA (Linear Discriminant Analysis)
## Dataset
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

![Dataset Visualized](visualization/fashion-mnist-sprite.png "Dataset Visualized")

## Tunable Parameters
### PCA Parameters
- **pca_target_dim:** Using PCA to reduce the data dimension to this number.

### LDA Parameters
- **components_number:** Using LDA to reduce the data dimension to this number.

### KNN Parameters
- **neighbor_num:** Number of neighbors taken into calculation.

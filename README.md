# Gaussian-Bayes and KNN on Fashion MNIST Dataset
[![Build Status](https://travis-ci.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset.svg?branch=master)](https://travis-ci.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)
[![GitHub](https://img.shields.io/github/license/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset/blob/master/LICENSE)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset?include_prereleases)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset/releases)
[![GitHub repo size](https://img.shields.io/github/repo-size/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://github.com/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bff26b8d5c544e84a70ca430d1129d57)](https://www.codacy.com/manual/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset&amp;utm_campaign=Badge_Grade)
[![GitHub top language](https://img.shields.io/github/languages/top/nuclearczy/Gaussian-Bayes_and_KNN_on_Fashion_MNIST_Dataset)](https://www.python.org/)

**Author: Zuyang Cao**

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

*Figure 1. Visualized Dataset*

## Tunable Parameters

### PCA Parameters
- **pca_target_dim:** Using PCA to reduce the data dimension to this number.

### LDA Parameters
- **components_number:** Number of components (< n_classes - 1) for dimensionality reduction.

### KNN Parameters
- **neighbor_num:** Number of neighbors taken into calculation.

## Results

### KNN with Different Parameters

- K-Neighbors

![Accuracy vs K Neighbors_scaled](visualization/KNN%20Accuracy%20vs%20K%20Neighbors.png)

*Figure 2. Accuracy and K Number*

From figure 2, it is clear that KNN reaches 100% accuracy on training set when K is set to 1. This is a typical 
overfitting circumstance. When increasing the K number, the accuracy on test set increased slightly and begin to be 
stable after K reaches 7. 

- Dimension Reduction Parameters

![Accuracy vs PCA&LDA](visualization/KNN%20Accuracy%20vs%20PCA_LDA%20N%20Dimensions.png)

*Figure 3. Accuracy with PCA and LDA*

![Low PCA&LDA Parameters](visualization/KNN%20Accuracy%20vs%20PCA_LDA%20N%20Dimensions_Low.png)

*Figure 4. Accuracy with Low PCA and LDA Value*

### Bayes vs KNN
The gaussian based Bayes classifier is a simple self built class, thus the accuracy maybe lower than the built-in 
classifier from scikit-learn or other libraries.

PCA dimension is set to 30 and LDA set to default in both methods.

Datasets | Bayes Accuracy | KNN Accuracy
-------- | -------------- | ------------ 
LDA Training set |      |  
PCA Training set |      |
LDA Testing set | |
PCA Testing set | |


import matplotlib
import numpy as np
from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *


def dimension_visualize_2d():
    train_image_dim, train_label_dim = load_mnist('data/fashion', kind='train')
    test_image_dim, test_label_dim = load_mnist('data/fashion', kind='t10k')
    pca_train, pca_test = pca_reduction(train_image_dim, test_image_dim, pca_target_dim=2)
    lda_train, lda_test = lda_reduction(train_image_dim, train_label_dim, test_image_dim, components_number=2)
    

if __name__ == '__main__':
    dimension_visualize_2d()

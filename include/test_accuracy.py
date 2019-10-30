import numpy as np


def test_accuracy(predicted_cat, labeled_cat):
    predicted_cat_arr = np.array(predicted_cat)
    labeled_cat_arr = np.array(labeled_cat)
    accuracy = np.mean((predicted_cat_arr == labeled_cat_arr) * 1)
    return accuracy

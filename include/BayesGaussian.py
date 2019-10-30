import numpy as np
from scipy.stats import multivariate_normal
import time


class BayesGaussian:
    def __init__(self):
        self.fit_data = dict()
        self.category_num = 0
        self.train_data = np.mat([])
        self.train_label = np.mat([])
        self.test_data = np.mat([])
        self.predict_label = np.mat([])
        self.test_categories = []

    def gaussian_fit(self, category_i):
        indexes = [i for i, label in enumerate(self.train_label) if label == category_i]
        class_i_images = []
        for i in indexes:
            class_i_images.append(self.train_data[i])
        i_images_matrix = np.array(class_i_images)
        class_i_mean = i_images_matrix.mean(0)
        class_i_cov = np.cov(i_images_matrix, rowvar=0)
        self.fit_data[f"mean{category_i}"] = class_i_mean
        self.fit_data[f"cov{category_i}"] = class_i_cov

    def fit(self, input_train, input_label):
        start_time = time.time()
        print("\nGaussian fitting in progress >>> ")
        self.train_data = input_train
        self.train_label = input_label
        self.category_num = max(input_label)
        for i in range(self.category_num + 1):
            self.gaussian_fit(i)
        end_time = time.time()
        print("Gaussian fit time : ", end_time - start_time, " seconds. ")
        print(">>> Done Gaussian fitting\n")

    def gaussian_predict(self, test_data):
        start_time = time.time()
        self.test_data = test_data
        print("\nGaussian predicting in progress >>> ")
        for test_sample in self.test_data:
            category_possibilities = []
            for i in range(self.category_num):
                i_possibility = multivariate_normal.pdf(test_sample, mean=self.fit_data[f"mean{i}"],
                                                        cov=self.fit_data[f"cov{i}"])
                category_possibilities.append(i_possibility)
            self.test_categories.append(np.argmax(category_possibilities))
        end_time = time.time()
        print("Gaussian predicting time : ", end_time - start_time, " seconds. ")
        print(">>> Done Gaussian predicting\n")
        return self.test_categories

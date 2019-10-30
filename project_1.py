import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time
from scipy.stats import multivariate_normal
from utils.mnist_reader import *


def pca_reduction(input_train, input_test, pca_target_dim=0):
    start_time = time.time()
    print("\nPCA in progress >>> ")
    if pca_target_dim:
        pca = PCA(n_components=pca_target_dim)
        print("PCA target dimension chosen as: ", pca.n_components)
    else:
        pca = PCA()
        print("PCA target dimension selected as auto")
    pca.fit(input_train)
    pca_train = pca.transform(input_train)
    pca_test = pca.transform(input_test)
    end_time = time.time()
    print("PCA time : ", end_time - start_time, " seconds. ")
    print(">>> Done PCA\n")
    return pca_train, pca_test


def lda_reduction(input_train, input_train_label, input_test, components_number=None):
    start_time = time.time()
    print("\nLDA in progress >>> ")
    lda = LDA(n_components=components_number)
    lda.fit(input_train, input_train_label)
    lda_train = lda.transform(input_train)
    lda_test = lda.transform(input_test)
    end_time = time.time()
    print("LDA time : ", end_time - start_time, " seconds. ")
    print(">>> Done LDA\n")
    return lda_train, lda_test


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
        # print(indexes)
        for i in indexes:
            class_i_images.append(self.train_data[i])
        i_images_matrix = np.array(class_i_images)
        class_i_mean = i_images_matrix.mean(0)
        # print(np.shape(class_i_mean))
        # print("mean : ", class_i_mean)
        # class_i_cov = np.cov(i_images_matrix.T)
        class_i_cov = np.cov(i_images_matrix, rowvar=0)
        # print(np.shape(class_i_cov))
        # print(np.linalg.det(class_i_cov))
        # print("cov : ", class_i_cov)
        self.fit_data[f"mean{category_i}"] = class_i_mean
        self.fit_data[f"cov{category_i}"] = class_i_cov

    def fit(self, input_train, input_label):
        start_time = time.time()
        print("\nGaussian fitting in progress >>> ")
        self.train_data = input_train
        self.train_label = input_label
        self.category_num = max(input_label)
        # print(self.train_data[1])
        for i in range(self.category_num + 1):
            self.gaussian_fit(i)
        # possibility = multivariate_normal.pdf(self.train_data[3], mean=self.fit_data[f"mean{0}"],
        #                                       cov=self.fit_data[f"cov{0}"])
        # print(possibility)
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

def test_accuracy(predicted_cat, labeled_cat):
    predicted_cat_arr = np.array(predicted_cat)
    labeled_cat_arr = np.array(labeled_cat)
    accuracy = np.mean((predicted_cat_arr == labeled_cat_arr) * 1)
    return accuracy

train_image, train_label = load_mnist('data/fashion', kind='train')
test_image, test_label = load_mnist('data/fashion', kind='t10k')
# print(train_label)
# print(np.shape(train_image[0]))
# print(train_image[1])
# image0 = np.mat(train_image[10])
# cv2.imshow("test", image0.reshape((28, 28)))
# cv2.waitKey(0)
pca_train, pca_test = pca_reduction(train_image, test_image, pca_target_dim=50)
lda_train, lda_test = lda_reduction(train_image, train_label, test_image)
# print(np.shape(pca_train))
# print(pca_train[0])
# print(np.shape(lda_train))
# print(lda_train[0])
bayes = BayesGaussian()
bayes.fit(lda_train, train_label)
predict_category = bayes.gaussian_predict(lda_test)
accuracy = test_accuracy(predict_category, test_label)
print(accuracy)
bayes2 = BayesGaussian()
bayes2.fit(pca_train, train_label)
predict_category_pca = bayes2.gaussian_predict(pca_test)
accuracy = test_accuracy(predict_category_pca, test_label)
print(accuracy)

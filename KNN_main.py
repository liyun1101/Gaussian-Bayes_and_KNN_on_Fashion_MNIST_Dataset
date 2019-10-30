from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *
from include.test_accuracy import *
from include.knn_predict import *


def main():
    train_image, train_label = load_mnist('data/fashion', kind='train')
    test_image, test_label = load_mnist('data/fashion', kind='t10k')
    pca_train, pca_test = pca_reduction(train_image, test_image, pca_target_dim=30)
    lda_train, lda_test = lda_reduction(train_image, train_label, test_image)

    predict_category_lda = knn_predict(lda_train, train_label, lda_test)
    accuracy_lda = test_accuracy(predict_category_lda, test_label)
    print("KNN accuracy with LDA dataset: ", np.round(accuracy_lda * 100, decimals=2), "%")

    predict_category_pca = knn_predict(pca_train, train_label, pca_test)
    accuracy_pca = test_accuracy(predict_category_pca, test_label)
    print("KNN accuracy with PCA dataset: ", np.round(accuracy_pca * 100, decimals=2), "%")


if __name__ == '__main__':
    main()

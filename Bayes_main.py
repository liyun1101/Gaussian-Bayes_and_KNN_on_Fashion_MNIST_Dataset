from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *
from include.BayesGaussian import *
from include.test_accuracy import *


def main():
    train_image, train_label = load_mnist('data/fashion', kind='train')
    test_image, test_label = load_mnist('data/fashion', kind='t10k')
    pca_train, pca_test = pca_reduction(train_image, test_image, pca_target_dim=50)
    lda_train, lda_test = lda_reduction(train_image, train_label, test_image)
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


if __name__ == '__main__':
    main()

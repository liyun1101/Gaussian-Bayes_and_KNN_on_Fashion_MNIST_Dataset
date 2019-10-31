from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *
from include.BayesGaussian import *
from include.test_accuracy import *


def bayes_main(pca_dim=30, lda_comp=None):
    train_image, train_label = load_mnist('data/fashion', kind='train')
    test_image, test_label = load_mnist('data/fashion', kind='t10k')
    pca_train, pca_test = pca_reduction(train_image, test_image, pca_target_dim=pca_dim)
    lda_train, lda_test = lda_reduction(train_image, train_label, test_image, components_number=lda_comp)

    bayes_lda_train = BayesGaussian()
    bayes_lda_train.fit(lda_train, train_label)
    predict_category_lda_train = bayes_lda_train.gaussian_predict(lda_train)
    accuracy_lda_train = test_accuracy(predict_category_lda_train, train_label)
    print("Bayes accuracy with training LDA dataset: ", np.round(accuracy_lda_train * 100, decimals=2), "%")

    bayes_pca_train = BayesGaussian()
    bayes_pca_train.fit(pca_train, train_label)
    predict_category_pca_train = bayes_pca_train.gaussian_predict(pca_train)
    accuracy_pca_train = test_accuracy(predict_category_pca_train, train_label)
    print("Bayes accuracy with training PCA dataset: ", np.round(accuracy_pca_train * 100, decimals=2), "%")

    bayes_lda = BayesGaussian()
    bayes_lda.fit(lda_train, train_label)
    predict_category_lda = bayes_lda.gaussian_predict(lda_test)
    accuracy_lda = test_accuracy(predict_category_lda, test_label)
    print("Bayes accuracy with LDA dataset: ", np.round(accuracy_lda * 100, decimals=2), "%")

    bayes_pca = BayesGaussian()
    bayes_pca.fit(pca_train, train_label)
    predict_category_pca = bayes_pca.gaussian_predict(pca_test)
    accuracy_pca = test_accuracy(predict_category_pca, test_label)
    print("Bayes accuracy with PCA dataset: ", np.round(accuracy_pca * 100, decimals=2), "%")
    return


if __name__ == '__main__':
    bayes_main()

from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *
from include.test_accuracy import *
from include.knn_predict import *


def knn_main(pca_dim=30, lda_comp=None, neighbor_num=5, weight='uniform'):
    train_image_knn, train_label_knn = load_mnist('data/fashion', kind='train')
    test_image_knn, test_label_knn = load_mnist('data/fashion', kind='t10k')
    pca_train_knn, pca_test_knn = pca_reduction(train_image_knn, test_image_knn, pca_target_dim=pca_dim)
    lda_train_knn, lda_test_knn = lda_reduction(train_image_knn, train_label_knn, test_image_knn,
                                                components_number=lda_comp)

    predict_category_lda = knn_predict(lda_train_knn, train_label_knn, lda_train_knn, num_of_neighbors=neighbor_num,
                                       weight_ctrl=weight)
    accuracy_lda_train = test_accuracy(predict_category_lda, train_label_knn)
    print("KNN training set accuracy with LDA dataset: ", np.round(accuracy_lda_train * 100, decimals=2), "%")

    predict_category_pca = knn_predict(pca_train_knn, train_label_knn, pca_train_knn, num_of_neighbors=neighbor_num,
                                       weight_ctrl=weight)
    accuracy_pca_train = test_accuracy(predict_category_pca, train_label_knn)
    print("KNN training accuracy with PCA dataset: ", np.round(accuracy_pca_train * 100, decimals=2), "%")

    predict_category_lda = knn_predict(lda_train_knn, train_label_knn, lda_test_knn, num_of_neighbors=neighbor_num,
                                       weight_ctrl=weight)
    accuracy_lda = test_accuracy(predict_category_lda, test_label_knn)
    print("KNN accuracy with LDA dataset: ", np.round(accuracy_lda * 100, decimals=2), "%")

    predict_category_pca = knn_predict(pca_train_knn, train_label_knn, pca_test_knn, num_of_neighbors=neighbor_num,
                                       weight_ctrl=weight)
    accuracy_pca = test_accuracy(predict_category_pca, test_label_knn)
    print("KNN accuracy with PCA dataset: ", np.round(accuracy_pca * 100, decimals=2), "%")

    return accuracy_lda_train, accuracy_pca_train, accuracy_lda, accuracy_pca


if __name__ == '__main__':
    knn_main()

import matplotlib.pyplot as plt
import KNN_main


def knn_visualize():
    # Test K Neighbors difference
    accuracy_lda_train_list = []
    accuracy_pca_train_list = []
    accuracy_lda_list = []
    accuracy_pca_list = []
    k_list = list(range(1, 21))
    for k_neighbor in range(1, 21):
        print(f"\n***** K = {k_neighbor} *****\n")
        accuracy_lda_train, accuracy_pca_train, accuracy_lda, accuracy_pca = KNN_main.knn_main(neighbor_num=k_neighbor)
        accuracy_lda_train_list.append(accuracy_lda_train)
        accuracy_pca_train_list.append(accuracy_pca_train)
        accuracy_lda_list.append(accuracy_lda)
        accuracy_pca_list.append(accuracy_pca)
    plt.plot(k_list, accuracy_lda_train_list, 'r--', label='LDA on Training Set')
    plt.plot(k_list, accuracy_pca_train_list, 'g--', label='PCA on Training Set')
    plt.plot(k_list, accuracy_lda_list, 'b--', label='LDA on Testing Set')
    plt.plot(k_list, accuracy_pca_list, 'c--', label='PCA on Testing Set')
    plt.title('KNN Accuracy vs K Neighbors')
    plt.xlabel('K Neighbors')
    plt.ylabel('Accuracy')
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.savefig('./visualization/KNN Accuracy vs K Neighbors.png')
    plt.show()

    # # Test N dimensions difference
    # accuracy_lda_train_list = []
    # accuracy_pca_train_list = []
    # accuracy_lda_list = []
    # accuracy_pca_list = []
    # k_list = list(range(10, 41))
    # for n_dim in range(10, 41):
    #     print(f"\n***** K dim = {n_dim} *****\n")
    #     accuracy_lda_train, accuracy_pca_train, accuracy_lda, accuracy_pca = KNN_main.knn_main(pca_dim=n_dim,
    #                                                                                            lda_comp=n_dim)
    #     accuracy_lda_train_list.append(accuracy_lda_train)
    #     accuracy_pca_train_list.append(accuracy_pca_train)
    #     accuracy_lda_list.append(accuracy_lda)
    #     accuracy_pca_list.append(accuracy_pca)
    # plt.plot(k_list, accuracy_lda_train_list, 'r--', label='LDA on Training Set')
    # plt.plot(k_list, accuracy_pca_train_list, 'g--', label='PCA on Training Set')
    # plt.plot(k_list, accuracy_lda_list, 'b--', label='LDA on Testing Set')
    # plt.plot(k_list, accuracy_pca_list, 'c--', label='PCA on Testing Set')
    # plt.title('KNN Accuracy vs PCA&LDA N Dimensions')
    # plt.xlabel('N Dimensions')
    # plt.ylabel('Accuracy')
    # axes = plt.gca()
    # axes.set_ylim([0, 1])
    # plt.legend()
    # plt.savefig('./visualization/KNN Accuracy vs PCA_LDA N Dimensions.png')
    # plt.show()


if __name__ == '__main__':
    knn_visualize()

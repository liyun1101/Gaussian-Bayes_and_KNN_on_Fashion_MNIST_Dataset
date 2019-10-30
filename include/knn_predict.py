from sklearn.neighbors import KNeighborsClassifier
import time


def knn_predict(data_train, label_train, data_test, num_of_neighbors=5, weight_ctrl='uniform'):
    start_time = time.time()
    print("\nKNN prediction in progress >>> ")
    neigh = KNeighborsClassifier(n_neighbors=num_of_neighbors, weights=weight_ctrl, n_jobs=-1)
    neigh.fit(data_train, label_train)
    test_predict = neigh.predict(data_test)
    end_time = time.time()
    print("KNN prediction time : ", end_time - start_time, " seconds. ")
    print(">>> Done KNN prediction\n")
    return test_predict

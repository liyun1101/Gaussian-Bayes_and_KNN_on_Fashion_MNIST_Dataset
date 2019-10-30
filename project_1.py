from utils.mnist_reader import *
from include.pca_reduction import *
from include.lda_reduction import *
from include.BayesGaussian import *
from include.test_accuracy import *


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

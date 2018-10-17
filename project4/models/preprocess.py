import tensorflow as tf
import numpy as np

def load_data_from_keras():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
	Y_train = np.zeros((X_train.shape[0], 10))
	Y_train[np.arange(X_train.shape[0]),y_train.reshape((-1))] = 1
	Y_test = np.zeros((X_test.shape[0], 10))
	Y_test[np.arange(X_test.shape[0]),y_test.reshape((-1))] = 1
	return (X_train, Y_train), (X_test, Y_test)

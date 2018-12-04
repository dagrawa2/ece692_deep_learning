import pickle
import tensorflow as tf
import numpy as np

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def load_data():
	X_train = np.array([])
	y_train = np.array([])
	for i in range(1, 6):
		batch = unpickle("cifar10/cifar-10-batches-py/data_batch_"+str(i))
		X_train = np.concatenate((X_train, batch[b"data"]), axis=0)
		y_train = np.concatenate((y_train, np.asarray(batch[b"labels"])), axis=0)
	batch = unpickle("cifar10/cifar-10-batches-py/test_batch")
	X_test = batch[b"data"]
	y_train = np.asarray(batch[b"labels"])
	X_train = X_train.reshape((-1, 3, 32, 32)).transpose(axes=(0, 2, 3, 1))
	X_test = X_test.reshape((-1, 3, 32, 32)).transpose(axes=(0, 2, 3, 1))
	Y_train = np.zeros((X_train.shape[0], 10))
	Y_train[np.arange(X_train.shape[0]),y_train] = 1
	Y_test = np.zeros((X_test.shape[0], 10))
	Y_test[np.arange(X_test.shape[0]),y_test] = 1
	return (X_train, Y_train), (X_test, Y_test)

def load_data_from_keras():
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
	Y_train = np.zeros((X_train.shape[0], 10))
	Y_train[np.arange(X_train.shape[0]),y_train.reshape((-1))] = 1
	Y_test = np.zeros((X_test.shape[0], 10))
	Y_test[np.arange(X_test.shape[0]),y_test.reshape((-1))] = 1
	return (X_train, Y_train), (X_test, Y_test)

def resize(X):
	with tf.Session() as sess:
		X = sess.run(tf.image.resize_images(X, [224, 224]))
	return X

def center(X):
	return X - np.mean(X, axis=(0, 1, 2), keepdims=True)

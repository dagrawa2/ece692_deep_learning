import gpflow
import numpy as np
import tensorflow as tf

def load_fashion(n_classes=10, p=1, map_into_unit_range=False):
	# load data
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	# extract only fraction p of the total data
	# maintain class balance
	indices_train = np.arange(y_train.shape[0])
	indices_test = np.arange(y_test.shape[0])
	ind_train = np.array([])
	ind_test = np.array([])
	for c in range(n_classes):
		temp = indices_train[y_train==c]
		ind_train = np.concatenate((ind_train, temp[:int(p*len(temp))]))
		temp = indices_test[y_test==c]
		ind_test = np.concatenate((ind_test, temp[:int(p*len(temp))]))
	# ensure integer indices
	ind_train = ind_train.astype(np.int32)
	ind_test = ind_test.astype(np.int32)
	# this is fraction p of the original data
	X_train = X_train[ind_train]
	y_train = y_train[ind_train]
	X_test = X_test[ind_test]
	y_test = y_test[ind_test]
	# convert class vectors to binary class matrices
	Y_train = tf.keras.utils.to_categorical(y_train, n_classes)
	Y_test = tf.keras.utils.to_categorical(y_test, n_classes)
	# use gpflow data type
	X_train = X_train.astype(gpflow.settings.float_type)
	X_test = X_test.astype(gpflow.settings.float_type)
	Y_train = Y_train.astype(gpflow.settings.float_type)
	Y_test = Y_test.astype(gpflow.settings.float_type)
	# reshape inputs
	X_train = np.expand_dims(X_train, -1)
	X_test = np.expand_dims(X_test, -1)
	if map_into_unit_range:
		# bring pixels into unit range
		X_train = X_train/255
		X_test = X_test/255
	return (X_train, Y_train), (X_test, Y_test)

def load_mnist(n_classes=10, p=1, map_into_unit_range=False):
	# load data
	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
	# extract only fraction p of the total data
	# maintain class balance
	indices_train = np.arange(y_train.shape[0])
	indices_test = np.arange(y_test.shape[0])
	ind_train = np.array([])
	ind_test = np.array([])
	for c in range(n_classes):
		temp = indices_train[y_train==c]
		ind_train = np.concatenate((ind_train, temp[:int(p*len(temp))]))
		temp = indices_test[y_test==c]
		ind_test = np.concatenate((ind_test, temp[:int(p*len(temp))]))
	# ensure integer indices
	ind_train = ind_train.astype(np.int32)
	ind_test = ind_test.astype(np.int32)
	# this is fraction p of the original data
	X_train = X_train[ind_train]
	y_train = y_train[ind_train]
	X_test = X_test[ind_test]
	y_test = y_test[ind_test]
	# convert class vectors to binary class matrices
	Y_train = tf.keras.utils.to_categorical(y_train, n_classes)
	Y_test = tf.keras.utils.to_categorical(y_test, n_classes)
	# use gpflow data type
	X_train = X_train.astype(gpflow.settings.float_type)
	X_test = X_test.astype(gpflow.settings.float_type)
	Y_train = Y_train.astype(gpflow.settings.float_type)
	Y_test = Y_test.astype(gpflow.settings.float_type)
	# reshape inputs
	X_train = np.expand_dims(X_train, -1)
	X_test = np.expand_dims(X_test, -1)
	if map_into_unit_range:
		# bring pixels into unit range
		X_train = X_train/255
		X_test = X_test/255
	return (X_train, Y_train), (X_test, Y_test)

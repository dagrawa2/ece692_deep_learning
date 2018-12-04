import models
from models.layers import *
import numpy as np

np.random.seed(123)

print("Loading data . . . ")
(X_train, Y_train), (X_test, Y_test) = models.load_data_from_keras()

"""
n = 1000
X_train = X_train[:n]
Y_train = Y_train[:n]
X_test = X_test[:n]
Y_test = Y_test[:n]
"""

m = X_train.min()
s = X_train.max()
X_train -= m
X_train = X_train/s
X_test -= m
X_test = X_test/s

layers = [Corrupt(noise_stdev=0.1),
	Convolution2D([3, 3, 3, 64], activation=tf.nn.relu, scope='conv1_1'),
	Convolution2D([3, 3, 64, 64], activation=tf.nn.relu, scope='conv1_2'),
	MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool1'),
#	Corrupt(noise_stdev=0.1),
	Convolution2D([3, 3, 64, 128], activation=tf.nn.relu, scope='conv2_1'),
	Convolution2D([3, 3, 128, 128], activation=tf.nn.relu, scope='conv2_2'),
	Convolution2D([3, 3, 128, 128], activation=tf.nn.relu, scope='conv2_3'),
#	MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool2'),
#	Unfold(scope='unfold'),
#	Corrupt(),
#	FullyConnected(512, activation=tf.nn.relu, scope='fc1')
]

blocks = []

model = models.autoencoder(layers, blocks, lr=1e-3, lr_decay=0.1, mbs=50, pred_mbs=500, seed=456)
model.start_session()
print("Training . . . ")
losses, params = model.train(X_train, X_test, epochs=30, early_stopping=5)
print("Denoising . . . ")
X_train = model.denoise(X_train)
X_test = model.denoise(X_test)
model.stop_session()

print("Saving results . . . ")
np.save("results/vgg7_ae_losses.npy", losses)
np.savez("results/vgg7_ae_params.npz", **params)
np.save("results/vgg7_ae_X_train.npy", X_train)
np.save("results/vgg7_ae_X_test.npy", X_test)

print("Done!")

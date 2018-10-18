import models
import numpy as np

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
	Corrupt(noise_stdev=0.1),
	Convolution2D([3, 3, 64, 128], activation=tf.nn.relu, scope='conv2_1'),
	Convolution2D([3, 3, 128, 128], activation=tf.nn.relu, scope='conv2_2'),
	Convolution2D([3, 3, 128, 128], activation=tf.nn.relu, scope='conv2_3'),
	MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool2'),
	Unfold(scope='unfold'),
	Corrupt(),
	FullyConnected(512, activation=tf.nn.relu, scope='fc1')
]

blocks = [[0, 1, 2],
	[4, 5, 6, 7],
	[10, 11]
]


model = models.autoencoder(layers, blocks, lr=1e-4, mbs=50, pred_mbs=50)
print("Pretraining . . . ")
model.pretrain(X_train, epochs=5)
print("Training . . . ")
losses, params = model.train(X_train, X_test, epochs=25, early_stopping=5)

print("Saving results . . . ")
np.save("results/losses.npy", losses)
np.savez("results/params.npz", **params)

print("Done!")

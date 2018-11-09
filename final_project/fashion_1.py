import gpflow
import numpy as np
import tensorflow as tf

np.random.seed(123)
tf.set_random_seed(345)

from my_dkl import datasets
from my_dkl import models
from my_dkl.layers import *

(X_train, Y_train), (X_test, Y_test) = datasets.load_fashion(n_classes=2, p=0.1)

input_dim = list(X_train.shape[1:])
layers = [input(input_dim),
	Convolution2D([5, 5, 1, 32], activation=tf.nn.relu, name="conv1"),
	MaxPooling(name="pool1"),
	Convolution2D([5, 5, 32, 64], activation=tf.nn.relu, name="conv2"),
	MaxPooling(name="pool2"),
	Unfold(name="unfold"),
	FullyConnected([7*7*64, 1024], activation=tf.nn.relu, name="fc1"),
	FullyConnected([1024, 2], name="fc2")
]

model = models.NN(X_train, Y_train, layers, minibatch_size=50)
accs = model.train(eval_set=(X_test, Y_test), lr=1e-3, epochs=50)

np.save("results/fashion_1_accs.npy", accs)

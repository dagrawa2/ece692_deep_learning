import gpflow
import numpy as np
import tensorflow as tf

from my_dkl import datasets
from my_dkl import models
from my_dkl.layers import *

np.random.seed(123)
tf.set_random_seed(345)

print("Loading MNIST data . . . ")
(X_train, Y_train), (X_test, Y_test) = datasets.load_mnist(n_classes=10, p=0.2, map_into_unit_range=True)

print("Initializing NN layers . . . ")
input_dim = list(X_train.shape)[1:]
output_dim = Y_train.shape[1]
layers = [Input(input_dim),
	Convolution2D([5, 5, 1, 32], activation=tf.nn.relu, name="conv1"),
	MaxPooling(name="pool1"),
	Convolution2D([5, 5, 32, 64], activation=tf.nn.relu, name="conv2"),
	MaxPooling(name="pool2"),
	Unfold(name="unfold"),
	FullyConnected([7*7*64, 1024], activation=tf.nn.relu, name="fc1"),
	FullyConnected([1024, output_dim], name="fc2")
]

print("Initializing NN model . . . ")
model = models.NN(X_train, Y_train, layers, minibatch_size=50)
accs = model.train(eval_set=(X_test, Y_test), lr=1e-3, epochs=2)

print("Saving accuracies . . . ")
np.save("results/mnist_2_accs.npy", accs)
print("Saving model parameters . . . ")
model.save_params("results/mnist_2_params.npz")

print("Done!")
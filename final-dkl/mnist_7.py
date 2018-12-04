import gpflow
import numpy as np
import tensorflow as tf

from my_dkl import datasets
from my_dkl import models
from my_dkl import kernels
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
#	FullyConnected([7*7*64, 1024], activation=tf.nn.relu, name="fc1")
#	FullyConnected([1024, output_dim], name="fc2")
]

print("Initializing base and deep kernels . . . ")
base_kernel = gpflow.kernels.RBF(7*7*64, variance=1.0, lengthscales=1.0)
kernel = kernels.deep_kernel(base_kernel, layers)

print("Initializing DKL model . . . ")
model = models.GPSGD(X_train, Y_train, kernel, minibatch_size=100)
model.likelihood.variance = 0.01

print("Training model . . . ")
accs = model.train(eval_set=(X_test, Y_test), epochs=20)
print("Saving accuracies . . . ")
np.save("results/mnist_7_accs_4.npy", accs)

print("Saving model parameters . . . ")
model.save_params("results/mnist_7_params.npz")

print("Done!")
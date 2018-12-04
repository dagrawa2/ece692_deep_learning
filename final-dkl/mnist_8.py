import gpflow
import numpy as np
import tensorflow as tf

from my_dkl import datasets
from my_dkl import models
from my_dkl import kernels

np.random.seed(123)
tf.set_random_seed(345)

print("Loading MNIST data . . . ")
(X_train, Y_train), (X_test, Y_test) = datasets.load_mnist(n_classes=10, p=0.2, map_into_unit_range=True)

print("Flattening images . . . ")
X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

print("Initializing kernel . . . ")
dim_input = X_train.shape[1]
kernel = gpflow.kernels.RBF(dim_input, variance=1.0, lengthscales=1.0)

print("Initializing NNGP model . . . ")
model = models.GPSGD(X_train, Y_train, kernel, minibatch_size=100)
model.likelihood.variance = 0.01

print("Training model . . . ")
accs = model.train(eval_set=(X_test, Y_test), epochs=20)

print("Saving accuracies . . . ")
np.save("results/mnist_8_accs.npy", accs)

print("Saving model parameters . . . ")
model.save_params("results/mnist_8_params.npz")

print("Done!")
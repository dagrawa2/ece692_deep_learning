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
kernel = kernels.NNGP(dim_input, v_b=1.0, v_w=1.0, depth=1)

print("Initializing NNGP model . . . ")
model = models.GPSGD(X_train, Y_train, kernel, minibatch_size=100)
model.likelihood.variance = 0.01

print("Training model . . . ")
accs = model.train(eval_set=(X_test, Y_test), epochs=10)

print("Saving accuracies . . . ")
np.save("results/mnist_5_accs.npy", accs)

print("Saving model parameters . . . ")
model.save_params("results/mnist_5_params.npz")

print("Done!")
import gpflow
import numpy as np
import tensorflow as tf

from my_dkl import datasets
from my_dkl import models
from my_dkl import kernels
from my_dkl.layers import *

np.random.seed(123)
tf.set_random_seed(345)

(X_train, Y_train), (X_test, Y_test) = datasets.load_fashion(n_classes=2, p=0.1)

input_dim = list(X_train.shape[1:])
layers = [input(input_dim),
	Convolution2D([5, 5, 1, 32], activation=tf.nn.relu, name="conv1"),
	MaxPooling(name="pool1"),
	Convolution2D([5, 5, 32, 64], activation=tf.nn.relu, name="conv2"),
	MaxPooling(name="pool2"),
	Unfold(name="unfold"),
#	FullyConnected([7*7*64, 1024], activation=tf.nn.relu, name="fc1")
#	FullyConnected([1024, 2], name="fc2")
]

base_kernel = kernels.NNGP(7*7*64, v_b=1.0, v_w=1.0, depth=1)
kernel = kernels.deep_kernel(base_kernel, layers)

model = models.GPSGD(X_train, Y_train, kernel, minibatch_size=100)
model.likelihood.variance = 0.01

for param in model.parameters:
	param.trainable = False

model.likelihood.variance.trainable = True
model.kern.v_b.trainable = True
model.kern.v_w.trainable = True

accs = model.train(eval_set=(X_test, y_test), epochs=10)
np.save("results/fashion_3_accs_1.npy", accs)

for param in model.parameters:
	param.trainable = True

model.train(eval_set=(X_test, y_test), epochs=10)
np.save("results/fashion_3_accs_2.npy", accs)

model.save_params("results/fashion_3_params.npz")

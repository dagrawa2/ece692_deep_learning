import gpflow
import numpy as np
import tensorflow as tf

from my_dkl import datasets
from my_dkl import models
from my_dkl import kernels
from my_dkl import uq
from my_dkl.layers import *

np.random.seed(123)
tf.set_random_seed(345)

print("Loading MNIST Fashion data . . . ")
(X_train, Y_train), (X_test, Y_test) = datasets.load_fashion(n_classes=10, p=0.2, map_into_unit_range=True)

print("Initializing NN layers . . . ")
input_dim = list(X_train.shape)[1:]
output_dim = Y_train.shape[1]
layers = [Input(input_dim),
	Convolution2D([3, 3, 1, 32], activation=tf.nn.relu, name="conv1_1"),
	Convolution2D([3, 3, 32, 32], activation=tf.nn.relu, name="conv1_2"),
	MaxPooling(name="pool1"),
	Convolution2D([3, 3, 32, 64], activation=tf.nn.relu, name="conv2_1"),
	Convolution2D([3, 3, 64, 64], activation=tf.nn.relu, name="conv2_2"),
	Convolution2D([3, 3, 64, 64], activation=tf.nn.relu, name="conv2_3"),
	MaxPooling(name="pool2"),
	Unfold(name="unfold"),
#	FullyConnected([7*7*64, 1024], activation=tf.nn.relu, name="fc1"),
#	FullyConnected([1024, output_dim], name="fc2")
]

print("Computing initial length scale . . . ")
v = np.sum((X_train - np.mean(X_train, axis=0, keepdims=True))**2)/X_train.shape[0]
l_0 = 0.1*np.sqrt(v)
print(". . . l_0 = "+str(np.round(l_0, 5)))

print("Initializing base and deep kernels . . . ")
base_kernel = gpflow.kernels.RBF(7*7*64, variance=1.0, lengthscales=l_0)
kernel = kernels.deep_kernel(base_kernel, layers)

print("Initializing DKL model . . . ")
model = models.GPSGD(X_train, Y_train, kernel, minibatch_size=100)
model.likelihood.variance = 0.01

print("Training model . . . ")
accs = model.train(eval_set=(X_test, Y_test), lr=1e-3, epochs=25)

print("Predicting on test set . . . ")
preds, vars = model.predict_y(X_test)
classes_true = np.argmax(Y_test, axis=1)
classes = np.argmax(preds, axis=1)
confs = np.amax(preds, axis=1)/vars[:,0] # 1/(1+vars[:,0])

print("Saving model parameters . . . ")
model.save_params("results/f_2_params.npz")

print("Saving accuracies . . . ")
with open("results/f_2_accs.csv", "w") as fp:
	fp.write("epoch,acc\n")
	for i, a in enumerate(accs):
		fp.write(str(i+1)+","+str(a)+"\n")

print("Saving test predictions . . . ")
with open("results/f_2_preds.csv", "w") as fp:
	fp.write("true,pred,conf\n")
	for t, p, c in zip(classes_true, classes, confs):
		fp.write(str(t)+","+str(p)+","+str(c)+"\n")

print("Saving ARC . . . ")
rejs, accs = uq.ARC(classes_true, classes, confs)
with open("results/f_2_arc.csv", "w") as fp:
	fp.write("rej,acc\n")
	for r, a in zip(rejs, accs):
		fp.write(str(r)+","+str(a)+"\n")

print("Done!")
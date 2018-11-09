import gpflow
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from my_dkl import models
from my_dkl import kernels

# load boston house-prices data set
boston = load_boston()
X = boston.data
y = boston.target
y = y.reshape((-1, 1))

# normalize data
X = (X-np.mean(X, axis=0, keepdims=True))/np.std(X, axis=0, keepdims=True)

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

# input dimension
dim_input = X_train.shape[1]

# instantiate an NNGP kernel
# and pass the input dimension as an argument
kernel = kernels.NNGP(dim_input, depth=3)

# instantiate GPSGD regressor
model = models.GPSGD(X_train, y_train, kernel, minibatch_size=5)
# set the variance of the Gaussian likelihood (variance of additive noise)
model.likelihood.variance = 0.01

#model.kern.v_b.trainable = False
#model.kern.v_w.trainable = False

# by default, the kernel hyperparameters are optimized during training
# uncommenting the next line fixes the hyperparameters to their initial values
#model.feature.set_trainable(False)

# print hyperparameter shapes and values before training
#print("Model hyperparameters before training:\n")
#print(model)
#print("\n")

# train the model
model.train(eval_set=(X_test, y_test), lr=1e-3, epochs=10)

# print optimized hyperparameter shapes and values
#print("Model hyperparameters after training:\n")
print(model)
print("\n")

# compute predictive mean and variance on test set
predict_mean, predict_var = model.predict_y(X_test)

# compute MSE in prediction
print("MSE in prediction on test set:")
print(np.sum((predict_mean-y_test)**2)/y_test.shape[0])

# compare the first 10 test predictions with the actual
print("\n\nSome example test predictions:\n")
for i in range(10):
	print("Prediction: ", np.round(predict_mean[i][0], 5), " +- ", np.round(np.sqrt(predict_var[i][0]), 5))
	print("Actual: ", y_test[i][0], "\n")

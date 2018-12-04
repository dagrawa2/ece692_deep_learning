import numpy as np
from copy import deepcopy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import to_categorical
from models.mine_test import *

digits = load_digits()
X = digits.data
y = digits.target

X = X.reshape((-1, 64))
X -= X.min()
X /= X.max()

Y = to_categorical(y, 10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)


layers = [64, 30, 10]
activations = [logistic]*(len(layers)-2) + [logistic]

model = NN(layers, activations, learning_rate=0.1, mini_batch_size=100, lambda_1=0, lambda_2=0, random_state=456, force_square_error=False)
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=10, early_stopping=None)

np.save("grads.npy", np.array(model.grads))
np.save("num_grads.npy", np.array(model.num_grads))

print(accs[-1])
print("Done")

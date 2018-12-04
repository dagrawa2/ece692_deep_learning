import keras
import numpy as np
from copy import deepcopy
from models.mine import *
from models.data_augmentation import *

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = np.expand_dims(X_train, -1)
X_train, y_train = augment_data(X_train, y_train, augmentation_factor=1, use_random_rotation=True, use_random_shift=True, use_random_zoom=True)

X_train = X_train.reshape((-1, 28*28))
X_test = X_test.reshape((-1, 28*28))

X_train -= X_train.min()
X_train = X_train/X_train.max()
X_test -= X_test.min()
X_test = X_test/X_test.max()

Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

model = NN([784, 30, 10], [logistic, logistic], learning_rate=0.1, mini_batch_size=100, lambda_1=0.1, lambda_2=0, random_state=456, force_square_error=False)
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=100, early_stopping=10)

np.save("results/1-3a.npy", accs)

print("Done")

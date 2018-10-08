import models
import numpy as np

print("Loading data . . . ")
#(X_train, Y_train), (X_test, Y_test) = models.load_data()
(X_train, Y_train), (X_test, Y_test) = models.load_data_from_keras()

print(Y_train[:3])
import sys
sys.exit()

"""
n = 10
X_train = X_train[:n]
Y_train = Y_train[:n]
X_test = X_test[:n]
Y_test = Y_test[:n]
"""

print("Resizing . . . ")
X_train = models.preprocess.resize(X_train)
X_test = models.preprocess.resize(X_test)

#print("Centering . . . ")
mean = 1.711960792541503906
X_train -= mean
X_test -= mean

print("Training . . . ")
model = models.scratch.vgg16(lr=1e-3, mbs=100, pred_mbs=100)
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=100, early_stopping=10)

print("Saving results . . . ")
np.save("results/accs.npy", accs)

print("Done!")

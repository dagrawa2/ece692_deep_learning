import models
import numpy as np

print("Loading data . . . ")
(X_train, Y_train), (X_test, Y_test) = models.load_data_from_keras()

"""
n = 1000
X_train = X_train[:n]
Y_train = Y_train[:n]
X_test = X_test[:n]
Y_test = Y_test[:n]
"""

m = X_train.min()
s = X_train.max()
X_train -= m
X_train = X_train/s
X_test -= m
X_test = X_test/s

print("Training . . . ")
model = models.vgg16(lr=1e-4, mbs=50, pred_mbs=50, retrain_last_n_layers=2)
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=25, early_stopping=5)

print("Saving results . . . ")
np.save("results/vgg16_pretrained_accs.npy", accs)

print("Done!")

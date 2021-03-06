import models
import numpy as np

np.random.seed(123)

print("Loading data . . . ")
(_, Y_train), (_, Y_test) = models.load_data_from_keras()
X_train = np.load("results/vgg16_ae_X_train.npy")
X_test = np.load("results/vgg16_ae_X_test.npy")

"""
n = 1000
X_train = X_train[:n]
Y_train = Y_train[:n]
X_test = X_test[:n]
Y_test = Y_test[:n]
"""

"""
m = X_train.min()
s = X_train.max()
X_train -= m
X_train = X_train/s
X_test -= m
X_test = X_test/s
"""

print("Training . . . ")
model = models.vgg16(lr=1e-3, mbs=50, pred_mbs=500, seed=456)
model.start_session()
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=100, early_stopping=5)
model.stop_session()

print("Saving results . . . ")
np.save("results/vgg16_denoised_accs.npy", accs)

print("Done!")

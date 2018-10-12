import models
import numpy as np

np.random.seed(123)

print("Loading data . . . ")
#(X_train, Y_train), (X_test, Y_test) = models.load_data()
(X_train, Y_train), (X_test, Y_test) = models.load_data_from_keras()

"""
n = 1000
X_train = X_train[:n]
Y_train = Y_train[:n]
X_test = X_test[:n]
Y_test = Y_test[:n]
"""

print("Centering . . . ")
mean = np.array([[1.253069180468749977e+02, 1.229503941406249936e+02, 1.138653831835937495e+02]])
X_train = X_train - mean
X_test = X_test - mean

print("Training . . . ")
model = models.scratch1.vgg16(lr=1e-4, mbs=100, pred_mbs=2000)
accs = model.train(X_train, Y_train, eval_set=(X_test, Y_test), epochs=100, early_stopping=5)

print("Saving results . . . ")
np.save("results/scratch-1a.npy", accs)

print("Done!")

import models
import numpy as np

print("Loading data . . . ")
#(X_train, Y_train), (X_test, Y_test) = models.load_data()
(X_train, Y_train), _ = models.load_data_from_keras()

print("Getting center . . . ")
center = np.mean(X_train, axis=(0, 1, 2))

print("Writing results . . . ")
np.savetxt("results/center.txt", center)

print("Done")
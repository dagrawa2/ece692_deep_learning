import os
import numpy as np

files = os.listdir("results/")
files.sort()
files.remove("1-3b-grads.npy")

for f in files:
	print(f)
	acc = np.load("results/"+f)
	print("epochs: ", len(acc))
	print("acc: ", np.max(acc), "\n")

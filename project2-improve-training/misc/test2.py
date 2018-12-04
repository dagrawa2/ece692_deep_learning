import numpy as np

grads = np.load("grads.npy")
num_grads = np.load("num_grads.npy")

for g, ng in list(zip(grads, num_grads))[::10]:
#	print(np.round(g, 5), ", ", np.round(ng, 5))
	print(g, ", ", ng)

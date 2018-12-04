import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

def epoch(a):
	return np.arange(1, len(a)+1)

acc1 = np.load("results/1-1a.npy")
acc2 = np.load("results/1-1b.npy")
acc3 = np.load("results/1-1c.npy")

plt.figure()
plt.plot(epoch(acc1), acc1, color="red", label="Quadratic")
plt.plot(epoch(acc2), acc2, color="green", label="Cross-entropy")
plt.plot(epoch(acc3), acc3, color="blue", label="Log-likelihood")
plt.legend(title="Loss")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracies for Task 1.1")
plt.savefig("plots/1-1.png", bbox_inches='tight')

acc0 = np.load("results/1-1b.npy")
acc1 = np.load("results/1-2a.npy")
acc2 = np.load("results/1-2b.npy")
acc3 = np.load("results/1-2c.npy")

plt.figure()
plt.plot(epoch(acc0), acc0, color="black", label="None")
plt.plot(epoch(acc1), acc1, color="red", label="L2")
plt.plot(epoch(acc2), acc2, color="green", label="L1")
plt.plot(epoch(acc3), acc3, color="blue", label="L1 + augmentation")
plt.legend(title="Regularization")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracies for Task 1.2")
plt.savefig("plots/1-2.png", bbox_inches='tight')

acc0 = np.load("results/1-2c.npy")
acc1 = np.load("results/1-3a.npy")
acc2 = np.load("results/1-3b.npy")

plt.figure()
plt.plot(epoch(acc0), acc0, color="black", label="0")
plt.plot(epoch(acc1), acc1, color="red", label="1")
plt.plot(epoch(acc2), acc2, color="green", label="2")
plt.legend(title="Num. Hidden layers")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracies for Task 1.3")
plt.savefig("plots/1-3.png", bbox_inches='tight')

grads = np.load("results/1-3b-grads.npy")

plt.figure()
plt.plot(epoch(grads[0]), grads[0], color="red", label="First")
plt.plot(epoch(grads[1]), grads[1], color="blue", label="Second")
plt.legend(title="Hidden layer")
plt.xlabel("Epoch")
plt.ylabel("Gradient magnitude")
plt.title("Rate of change of hidden neurons in Task 1.3")
plt.savefig("plots/1-3-grads.png", bbox_inches='tight')

acc0 = np.load("results/1-3b.npy")
acc1 = np.load("results/1-3c-1.npy")
acc2 = np.load("results/1-3c-2.npy")
acc3 = np.load("results/1-3c-3.npy")

plt.figure()
plt.plot(epoch(acc0), acc0, color="black", label="0")
plt.plot(epoch(acc1), acc1, color="red", label="0.1")
plt.plot(epoch(acc2), acc2, color="green", label="0.3")
plt.plot(epoch(acc3), acc3, color="blue", label="0.5")
plt.legend(title="Dropout rate")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracies using dropout in Task 1.3")
plt.savefig("plots/1-3-dropout.png", bbox_inches='tight')

acc = np.load("results/2-1.npy")

plt.figure()
plt.plot(epoch(acc), acc, color="black")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracies for Task 2")
plt.savefig("plots/2-1.png", bbox_inches='tight')

print("Done!")

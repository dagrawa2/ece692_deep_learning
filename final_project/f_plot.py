import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

print("Plotting accuracies . . . ")
a_1 = pd.read_csv("results/f_1_accs.csv", usecols=["acc"]).values.reshape((-1))
a_2 = pd.read_csv("results/f_2_accs.csv", usecols=["acc"]).values.reshape((-1))
a_3 = pd.read_csv("results/f_3_accs.csv", usecols=["acc"]).values.reshape((-1))
e = np.arange(a_1.shape[0])+1

plt.figure()
plt.plot(e, a_1, color="red", label="NN")
plt.plot(e, a_2, color="green", label="NN+RBF")
plt.plot(e, a_3, color="blue", label="NN+NNGP")
plt.legend(title="Model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test accuracy on Fashion")
plt.savefig("plots/f_accs.png", box_inches="tight")

print("Plotting ARCs . . . ")
arc = pd.read_csv("results/f_1_arc.csv")
r = arc["rej"].values.reshape((-1))
a_1 = arc["acc"].values.reshape((-1))
a_2 = pd.read_csv("results/f_2_arc.csv", usecols=["acc"]).values.reshape((-1))
a_3 = pd.read_csv("results/f_3_arc.csv", usecols=["acc"]).values.reshape((-1))

plt.figure()
plt.plot(r, a_1, color="red", label="NN")
plt.plot(r, a_2, color="green", label="NN+RBF")
plt.plot(r, a_3, color="blue", label="NN+NNGP")
plt.legend(title="Model")
plt.xlabel("Rejection")
plt.ylabel("Accuracy")
plt.title("ARC on Fashion")
plt.savefig("plots/f_arc.png", box_inches="tight")

print("Done!")
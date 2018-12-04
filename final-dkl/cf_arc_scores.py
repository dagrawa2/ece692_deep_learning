import numpy as np
import pandas as pd
from my_dkl import uq

file = open("results/cf_arc_scores.txt", "w")
file.write("\\begin{tabular}{|c|c|c|} \\hline\n")
file.write("\\quad & NN & NN+RBF & NN+NNGP \\\\ \\hline\n")

file.write("CIFAR10")
for i in range(1, 4):
	data = pd.read_csv("results/c_"+str(i)+"_arc.csv")
	rejs = data["rej"].values.reshape((-1))
	accs = data["acc"].values.reshape((-1))
	score = uq.ARC_L1_score(rejs, accs)
	file.write(" & "+str(np.round(score, 5)))
file.write(" \\\\ \\hline\n")

file.write("Fashion")
for i in range(1, 4):
	data = pd.read_csv("results/f_"+str(i)+"_arc.csv")
	rejs = data["rej"].values.reshape((-1))
	accs = data["acc"].values.reshape((-1))
	score = uq.ARC_L1_score(rejs, accs)
	file.write(" & "+str(np.round(score, 5)))
file.write(" \\\\ \\hline\n")

file.write("\\end{tabular}")
file.close()

print("Done!")
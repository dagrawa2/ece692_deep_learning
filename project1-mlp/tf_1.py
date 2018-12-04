import pickle
import tensorflow as tf
import numpy as np
from tf_model import Network

widths = [8, 32, 128, 512]
lrs = [0.01, 0.05, 0.1]
epochs = 50
mini_batch_size = 100
early_stopping_threshold = 1e-5

file = open("results/tf_1.csv", "w")
file.write("width,lr,epochs,accuracy\n")

accs = np.zeros((len(lrs), len(widths)))
for i, h in enumerate(widths):
	for j, lr in enumerate(lrs):
		print("Training width=", h, " lr=", lr)
		nn = Network(h_widths=[h], lr=lr)
		eval_acc = nn.train(epochs=epochs, mini_batch_size=mini_batch_size, monitor_evaluation=True, early_stopping_threshold=early_stopping_threshold)
		accs[j, i] = eval_acc[-1]
		file.write(str(h)+","+str(lr)+","+str(len(eval_acc))+","+str(eval_acc[-1])+"\n")

file.close()

accs = np.flip(accs, axis=0)

D = {"widths": widths, "lrs": lrs, "accs": accs}
with open("results/tf_1.dict", "wb") as fp:
	pickle.dump(D, fp)

print("Done")

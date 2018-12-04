import pickle
import numpy as np
from mnist_loader import load_data_wrapper
from np_model import Network

train_data, valid_data, _ = load_data_wrapper()
#train_data = train_data[:200]
#valid_data = valid_data[:100]

widths = [8, 32, 128, 512]
lrs = [0.01, 0.05, 0.1]
epochs = 50
mini_batch_size = 100
early_stopping_threshold = 1e-5

file = open("results/np_1.csv", "w")
file.write("width,lr,epochs,accuracy\n")

accs = np.zeros((len(lrs), len(widths)))
for i, h in enumerate(widths):
	for j, lr in enumerate(lrs):
		print("Training width=", h, " lr=", lr)
		nn = Network([784, h, 10])
		_, eval_acc, _, _ = nn.SGD(train_data, epochs, mini_batch_size, lr, evaluation_data=valid_data, monitor_evaluation_accuracy=True, early_stopping_threshold=early_stopping_threshold)
		accs[j, i] = eval_acc[-1]
		file.write(str(h)+","+str(lr)+","+str(len(eval_acc))+","+str(eval_acc[-1])+"\n")

file.close()

accs = np.flip(accs, axis=0)

D = {"widths": widths, "lrs": lrs, "accs": accs}
with open("results/np_1.dict", "wb") as fp:
	pickle.dump(D, fp)

print("Done")

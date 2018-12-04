import pickle
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

with open("results/np_2.dict", "rb") as fp:
	D = pickle.load(fp)

widths = D["widths"]
lrs = D["lrs"]
accs = D["accs"]

vmin, midpoint = np.min(accs), np.mean(accs)

plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(accs, interpolation='nearest', cmap=plt.cm.hot, norm=MidpointNormalize(vmin=vmin, midpoint=midpoint))
plt.xlabel("Width")
plt.ylabel("LR")
plt.colorbar()
plt.xticks(np.arange(len(widths)), widths, rotation=45)
plt.yticks(np.flip(np.arange(len(lrs)), axis=0), lrs)
plt.title("Validation accuracy (numpy, two hidden layers)")
plt.savefig("plots/np_2.png", bbox_inches='tight')
#plt.show()

print("Done")

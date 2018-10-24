import numpy as np

losses = np.load("results/vgg7_ae_losses.npy")
print("VGG7 AE losses")
print("epochs: ", len(losses))
print("Final loss: ", losses[-1])
print("Epoch of min loss: ", losses.argmin())
print("Min loss: ", losses.min())
print("\n")

accs = np.load("results/vgg7_accs.npy")
print("VGG7")
print("epochs: ", len(accs))
print("max accuracy: ", accs.max())
print("\n")

accs = np.load("results/vgg7_denoised_accs.npy")
print("VGG7 with denoised input")
print("epochs: ", len(accs))
print("max accuracy: ", accs.max())
print("\n")

accs = np.load("results/vgg7_pretrained_accs.npy")
print("VGG7 with pretrained weights")
print("epochs: ", len(accs))
print("max accuracy: ", accs.max())

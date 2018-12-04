set -e

# https://www.cs.toronto.edu/~frossard/post/vgg16/

mkdir frossard
cd frossard
echo "Downloading . . . "
wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz


set -e

# http://www.cs.toronto.edu/~kriz/cifar.html

mkdir cifar10
cd ciphar10
echo "Downloading . . . "
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo "Extracting . . . "
tar -xzf cifar-10-python.tar.gz

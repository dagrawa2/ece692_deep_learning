echo "Downloading DCGAN-tensorflow repo . . . "
git clone https://github.com/carpedm20/DCGAN-tensorflow

echo "Downloading MNIST data set . . . "
cd DCGAN-tensorflow
python3 download.py mnist
cd ..

echo "Installing scripts with personal modifications . . . "
cd DCGAN-tensorflow
mv main.py main-original.py
mv model.py model-original.py
mv utils.py utils-original.py
cd ..
cp scripts/main.py DCGAN-tensorflow/main.py
cp scripts/model.py DCGAN-tensorflow/model.py
cp scripts/utils.py DCGAN-tensorflow/utils.py

echo "Done!"
set -e

echo "pretrained-1.py"
python3 pretrained-1.py 2> out-1.txt

echo "pretrained-2.py"
python3 pretrained-2.py 2> out-2.txt

echo "pretrained-3.py"
python3 pretrained-3.py 2> out-3.txt

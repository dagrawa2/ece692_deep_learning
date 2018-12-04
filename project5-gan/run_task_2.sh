nohup \
python3 DCGAN-tensorflow/main.py --dataset mnist --input_height=28 --output_height=28 \
	--data_dir="./DCGAN-tensorflow/data" \
	--checkpoint_dir="./results_task_2/checkpoint" \
	--sample_dir="./results_task_2/samples" \
	--train \
	--conditional \
	--epoch=25 \
	--learning_rate=0.0002 \
	--batch_size=64 \
	--generate_train_images=20 \

mv nohup.out results_task_2/out.txt

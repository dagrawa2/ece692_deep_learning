import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=10000, seed=123)

class Network(object):

	def __init__(self, h_widths=[10], lr=0.01, init_weight_range=[-0.01, 0.01]):
		self.h_widths = h_widths
		self.lr = lr
		self.init_weight_range = init_weight_range
		self.build_graph()

	def build_graph(self):
		self.x = tf.placeholder(tf.float32, [None, 784])
		W = tf.Variable(tf.random_uniform([784, self.h_widths[0]], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
		b = tf.Variable(tf.random_uniform([self.h_widths[0]], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
		a = tf.nn.sigmoid(tf.matmul(self.x, W) + b)
		for width_in, width_out in zip(self.h_widths[:-1], self.h_widths[1:]):
			W = tf.Variable(tf.random_uniform([width_in, width_out], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
			b = tf.Variable(tf.random_uniform([width_out], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
			a = tf.nn.sigmoid(tf.matmul(a, W) + b)
		W = tf.Variable(tf.random_uniform([self.h_widths[-1], 10], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
		b = tf.Variable(tf.random_uniform([10], minval=self.init_weight_range[0], maxval=self.init_weight_range[1]))
		self.y = tf.nn.softmax(tf.matmul(a, W) + b)
		self.y_ = tf.placeholder(tf.float32, [None, 10])
		self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
		self.train_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy)

	def train(self, epochs=1, mini_batch_size=100, monitor_evaluation=False, early_stopping_threshold=None):
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		evaluation_accuracy = []
		n_batches = mnist.train.num_examples//mini_batch_size if mnist.train.num_examples%mini_batch_size==0 else 1+mnist.train.num_examples//mini_batch_size
		for epoch in range(epochs):
			for _ in range(n_batches):
				batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
				self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
#			print("Epoch %s training complete" % str(epoch+1))
			if monitor_evaluation:
				accuracy = self.eval()
				evaluation_accuracy.append(accuracy)
#				print("Accuracy on evaluation data: {}".format(accuracy))
			if early_stopping_threshold is not None and epoch >= 1 and evaluation_accuracy[-1]-evaluation_accuracy[-2] <= early_stopping_threshold:
				evaluation_accuracy = evaluation_accuracy[:-1]
				break
		return evaluation_accuracy
		
	def eval(self):
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		return self.sess.run(accuracy, feed_dict={self.x: mnist.validation.images, self.y_: mnist.validation.labels})



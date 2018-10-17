import time
import tensorflow as tf
import numpy as np
from layers import *

class autoencoder:

	def __init__(self, noise_std=0.1, lr=0.01, lr_decay=1, mbs=1, pred_mbs=None):
		self.noise_std = noise_std
		self.lr = lr
		self.lr_decay = lr_decay
		self.mbs = mbs
		self.pred_mbs = pred_mbs
		self.build_graph()

	def architecture(self, input):
		self.parameters = []
		self.parameter_names = []

		# encoder

		# corrupt input
		corrupted = np.minimum(np.maximum(input + np.random.normal(0, self.noise_std, size=input.shape), 0), 1)

		# block 1
		conv1_1 = Convolution2D([3, 3, 3, 64], activation=tf.nn.relu, scope='conv1_1')(corrupted)
		conv1_2 = Convolution2D([3, 3, 64, 64], activation=tf.nn.relu, scope='conv1_2')(conv1_1)
		pool1 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool1')(conv1_2)
		self.parameters = self.parameters + [conv1_1.kernel, conv1_1.bias, conv1_2.kernel, conv1_2.bias]
		self.parameter_names = self.parameter_names + ["conv1_1_W", "conv1_1_b", "conv1_2_W", "conv1_2_b"]

		# block 2
		conv2_1 = Convolution2D([3, 3, 64, 128], activation=tf.nn.relu, scope='conv2_1')(pool1)
		conv2_2 = Convolution2D([3, 3, 128, 128], activation=tf.nn.relu, scope='conv2_2')(conv2_1)
		pool2 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool2')(conv2_2)
		self.parameters = self.parameters + [conv2_1.kernel, conv2_1.bias, conv2_2.kernel, conv2_2.bias]
		self.parameter_names = self.parameter_names + ["conv2_1_W", "conv2_1_b", "conv2_2_W", "conv2_2_b"]

		# block 3
		conv3_1 = Convolution2D([3, 3, 128, 256], activation=tf.nn.relu, scope='conv3_1')(pool2)
		conv3_2 = Convolution2D([3, 3, 256, 256], activation=tf.nn.relu, scope='conv3_2')(conv3_1)
		conv3_3 = Convolution2D([3, 3, 256, 256], activation=tf.nn.relu, scope='conv3_3')(conv3_2)
		pool3 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool3')(conv3_3)
		self.parameters = self.parameters + [conv3_1.kernel, conv3_1.bias, conv3_2.kernel, conv3_2.bias, conv3_3.kernel, conv3_3.bias]
		self.parameter_names = self.parameter_names + ["conv3_1_W", "conv3_1_b", "conv3_2_W", "conv3_2_b", "conv3_3_W", "conv3_3_b"]

		# block 4
		conv4_1 = Convolution2D([3, 3, 256, 512], activation=tf.nn.relu, scope='conv4_1')(pool3)
		conv4_2 = Convolution2D([3, 3, 512, 512], activation=tf.nn.relu, scope='conv4_2')(conv4_1)
		conv4_3 = Convolution2D([3, 3, 512, 512], activation=tf.nn.relu, scope='conv4_3')(conv4_2)
		pool4 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool4')(conv4_3)
		self.parameters = self.parameters + [conv4_1.kernel, conv4_1.bias, conv4_2.kernel, conv4_2.bias, conv4_3.kernel, conv4_3.bias]
		self.parameter_names = self.parameter_names + ["conv4_1_W", "conv4_1_b", "conv4_2_W", "conv4_2_b", "conv4_3_W", "conv4_3_b"]

		# block 5
		conv5_1 = Convolution2D([3, 3, 512, 512], activation=tf.nn.relu, scope='conv5_1')(pool4)
		conv5_2 = Convolution2D([3, 3, 512, 512], activation=tf.nn.relu, scope='conv5_2')(conv5_1)
		conv5_3 = Convolution2D([3, 3, 512, 512], activation=tf.nn.relu, scope='conv5_3')(conv5_2)
		pool5 = MaxPooling(kernel_shape=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', scope='pool5')(conv5_3)
		self.parameters = self.parameters + [conv5_1.kernel, conv5_1.bias, conv5_2.kernel, conv5_2.bias, conv5_3.kernel, conv5_3.bias]
		self.parameter_names = self.parameter_names + ["conv5_1_W", "conv5_1_b", "conv5_2_W", "conv5_2_b", "conv5_3_W", "conv5_3_b"]

		# block 6
		unfold = Unfold(scope='unfold')(pool5)
		self.encoded = FullyConnected(512, activation=tf.nn.relu, scope='encode')(unfold)
		self.parameters = self.parameters + [self.encoded.weights, self.encoded.bias]
		self.parameter_names = self.parameter_names + ["fc1_W", "fc1_b"]

		# decoder

		# block 7
		decoded = FullyConnected(1*1*512, weights=tf.transpose(self.encoded.weights), activation=tf.nn.relu, scope='decode')(self.encoded)
		fold = Fold([-1, 1, 1, 512], scope='fold')(decoded)

		# block 8
		unpool5 = UnPooling((2, 2), output_shape=tf.shape(conv5_3), scope='unpool5')(fold)
		deconv5_3 = DeConvolution2D([3, 3, 512, 512], kernel=conv5_3.kernel, output_shape=tf.shape(conv5_2), activation=tf.nn.relu, scope='deconv5_3')(unpool5)
		deconv5_2 = DeConvolution2D([3, 3, 512, 512], kernel=conv5_2.kernel, output_shape=tf.shape(conv5_1), activation=tf.nn.relu, scope='deconv5_2')(deconv5_3)
		deconv5_1 = DeConvolution2D([3, 3, 512, 512], kernel=conv5_1.kernel, output_shape=tf.shape(pool4), activation=tf.nn.relu, scope='deconv5_1')(deconv5_2)

		# block 9
		unpool4 = UnPooling((2, 2), output_shape=tf.shape(conv4_3), scope='unpool4')(deconv5_1)
		deconv4_3 = DeConvolution2D([3, 3, 512, 512], kernel=conv4_3.kernel, output_shape=tf.shape(conv4_2), activation=tf.nn.relu, scope='deconv4_3')(unpool4)
		deconv4_2 = DeConvolution2D([3, 3, 512, 512], kernel=conv4_2.kernel, output_shape=tf.shape(conv4_1), activation=tf.nn.relu, scope='deconv4_2')(deconv4_3)
		deconv4_1 = DeConvolution2D([3, 3, 256, 512], kernel=conv4_1.kernel, output_shape=tf.shape(pool3), activation=tf.nn.relu, scope='deconv4_1')(deconv4_2)

		# block 10
		unpool3 = UnPooling((2, 2), output_shape=tf.shape(conv3_3), scope='unpool3')(deconv4_1)
		deconv3_3 = DeConvolution2D([3, 3, 256, 256], kernel=conv3_3.kernel, output_shape=tf.shape(conv3_2), activation=tf.nn.relu, scope='deconv3_3')(unpool3)
		deconv3_2 = DeConvolution2D([3, 3, 256, 256], kernel=conv3_2.kernel, output_shape=tf.shape(conv3_1), activation=tf.nn.relu, scope='deconv3_2')(deconv3_3)
		deconv3_1 = DeConvolution2D([3, 3, 128, 256], kernel=conv3_1.kernel, output_shape=tf.shape(pool2), activation=tf.nn.relu, scope='deconv3_1')(deconv3_2)

		# block 11
		unpool2 = UnPooling((2, 2), output_shape=tf.shape(conv2_2), scope='unpool2')(deconv3_1)
		deconv2_2 = DeConvolution2D([3, 3, 128, 128], kernel=conv2_2.kernel, output_shape=tf.shape(conv2_1), activation=tf.nn.relu, scope='deconv2_2')(deconv2_2)
		deconv2_1 = DeConvolution2D([3, 3, 64, 128], kernel=conv2_1.kernel, output_shape=tf.shape(pool1), activation=tf.nn.relu, scope='deconv2_1')(deconv2_2)

		# block 12
		unpool1 = UnPooling((2, 2), output_shape=tf.shape(conv1_2), scope='unpool1')(deconv2_1)
		deconv1_2 = DeConvolution2D([3, 3, 64, 64], kernel=conv1_2.kernel, output_shape=tf.shape(conv1_1), activation=tf.nn.relu, scope='deconv1_2')(deconv1_2)
		self.reconstruction = DeConvolution2D([3, 3, 3, 64], kernel=conv1_1.kernel, output_shape=tf.shape(input), activation=tf.nn.sigmoid, scope='deconv1_1')(deconv1_2)

	def build_graph(self):
		self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
		self.architecture()
		self.loss = tf.nn.l2_loss(self.x - self.reconstruction)
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def train(self, X_train, X_test, epochs=1, early_stopping=None):
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)
		if self.pred_mbs is None: self.pred_mbs = X_test.shape[0]
		n_train = X_train.shape[0]
		n_batches = int(np.ceil(n_train/self.mbs))
		mb_progress = int(0.1*n_batches)
		indices = np.arange(X_train.shape[0])
		losses = []
		best_parameters = self.sess.run(self.parameters)
		if early_stopping is None: early_stopping = epochs+1
		for epoch in range(epochs):
			print("Epoch ", epoch, " . . . ")
			time_0 = time.time()
			np.random.shuffle(indices)
			X = X_train[indices]
			if (1+epoch)%10 == 0: self.lr *= self.lr_decay
			time_1 = time.time()
			for k, i in enumerate(range(0,n_train,self.mbs)):
#				if k%mb_progress == 0:
#					print(k, "/", n_batches, " batches (", np.round(time.time()-#time_1, 5), " s)")
#					time_1 = time.time()
				X_batch = X[i:min(n_train,i+self.mbs)]
				self.sess.run([self.train_step], feed_dict={self.x: X_batch})
			losses.append(self.test_loss(X_test))
			print("\t Test loss: ", np.round(losses[-1], 6), " (", np.round(time.time()-time_0, 5), " s)")
			if 1+epoch >= 2 and losses[-1] < losses[-2]:
				best_parameters = self.sess.run(self.parameters)
			if 1+epoch >= early_stopping and np.argmin(lossess[-early_stopping:])==0: break
		sess.close()
		best_parameters = {name: value for name, value in zip(self.parameter_names, best_parameters)}
		return np.array(losses), best_parameters

	def test_loss(self, X):
		n_test = X.shape[0]
		batch_losses = []
		for i in range(0,n_test,self.pred_mbs):
			X_batch = X[i:min(n_test,i+self.pred_mbs)]
			batch_losses.append( self.sess.run(self.loss, feed_dict={self.x: X_batch}) )
		return np.sum(batch_losses)/n_test

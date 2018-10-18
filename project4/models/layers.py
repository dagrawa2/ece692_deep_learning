from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np


class Layer(object, metaclass=ABCMeta):
	"""

	"""
	def __init__(self):
		pass

	@abstractmethod
	def call(self, *args, **kwargs):
		raise NotImplementedError

	def __call__(self, *args, **kwargs):
		return self.call(*args, **kwargs)


class Convolution2D(Layer):
	"""

	"""
	def __init__(self,
				 kernel_shape,
				 kernel=None,
				 bias=None,
				 strides=(1, 1, 1, 1),
				 padding='SAME',
				 activation=None,
				 scope=''):
		Layer.__init__(self)

		self.kernel_shape = kernel_shape
		self.kernel = kernel
		self.bias = bias
		self.strides = strides
		self.padding = padding
		self.activation = activation
		self.scope = scope

	def build(self, input_tensor):
		# build kernel
		if self.kernel is not None:
			assert self.kernel.get_shape() == self.kernel_shape
		else:
			self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

		# build bias
		kernel_height, kernel_width, num_input_channels, num_output_channels = self.kernel.get_shape()
		if self.bias:
			assert self.bias.get_shape() == (num_output_channels, )
		else:
			self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

		# convolution
		conv = tf.nn.conv2d(input_tensor, self.kernel, strides=self.strides, padding=self.padding)

		# activation
		if self.activation:
			return self.activation(conv + self.bias)
		return conv + self.bias

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)

	def decoder(self, output_shape, activation):
		return DeConvolution2D(self.kernel_shape, output_shape, kernel=self.kernel, activation=activation, scope=self.scope+"_decoder")

	def get_params(self):
		return [self.kernel, self.bias]

	def get_param_names(self):
		return [self.scope+"_W", self.scope+"_b"]



class DeConvolution2D(Layer):
	"""

	"""
	def __init__(self,
				 kernel_shape,
				 output_shape,
				 kernel=None,
				 bias=None,
				 strides=(1, 1, 1, 1),
				 padding='SAME',
				 activation=None,
				 scope=''):
		Layer.__init__(self)

		self.kernel_shape = kernel_shape
		self.output_shape = output_shape
		self.kernel = kernel
		self.bias = bias
		self.strides = strides
		self.padding = padding
		self.activation = activation
		self.scope = scope

	def build(self, input_tensor):
		# build kernel
		if self.kernel:
			assert self.kernel.get_shape() == self.kernel_shape
		else:
			self.kernel = tf.Variable(tf.truncated_normal(self.kernel_shape, stddev=0.1), name='kernel')

		# build bias
		window_height, window_width, num_output_channels, num_input_channels = self.kernel.get_shape()
		if self.bias:
			assert self.bias.get_shape() == (num_output_channels, )
		else:
			self.bias = tf.Variable(tf.constant(0.1, shape=[num_output_channels]), name='bias')

		# convolution
		deconv = tf.nn.conv2d_transpose(input_tensor,
										self.kernel,
										output_shape=self.output_shape,
										strides=self.strides,
										padding=self.padding)

		# activation
		if self.activation:
			return self.activation(deconv + self.bias)
		return deconv + self.bias

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)


class MaxPooling(Layer):
	"""

	"""
	def __init__(self,
				 kernel_shape,
				 strides,
				 padding,
				 scope=''):
		Layer.__init__(self)

		self.kernel_shape = kernel_shape
		self.strides = strides
		self.padding = padding
		self.scope = scope

	def build(self, input_tensor):
		return tf.nn.max_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)

	def decoder(self, output_shape, activation):
		return UnPooling((2, 2), output_shape=output_shape, scope=self.scope+"_decoder")

	def get_params(self):
		return []

	def get_param_names(self):
		return []


class UnPooling(Layer):
	"""
	Unpool a max-pooled layer.

	Currently this method does not use the argmax information from the previous pooling layer.
	Currently this method assumes that the size of the max-pooling filter is same as the strides.

	Each entry in the pooled map would be replaced with an NxN kernel with the original entry in the upper left.
	For example: a 1x2x2x1 map of

		[[[[1], [2]],
		  [[3], [4]]]]

	could be unpooled to a 1x4x4x1 map of

		[[[[ 1.], [ 0.], [ 2.], [ 0.]],
		  [[ 0.], [ 0.], [ 0.], [ 0.]],
		  [[ 3.], [ 0.], [ 4.], [ 0.]],
		  [[ 0.], [ 0.], [ 0.], [ 0.]]]]
	"""
	def __init__(self,
				 kernel_shape,
				 output_shape,
				 scope=''):
		Layer.__init__(self)

		self.kernel_shape = kernel_shape
		self.output_shape = output_shape
		self.scope = scope

	def build(self, input_tensor):
		num_channels = input_tensor.get_shape()[-1]
		input_dtype_as_numpy = input_tensor.dtype.as_numpy_dtype()
		kernel_rows, kernel_cols = self.kernel_shape

		# build kernel
		kernel_value = np.zeros((kernel_rows, kernel_cols, num_channels, num_channels), dtype=input_dtype_as_numpy)
		kernel_value[0, 0, :, :] = np.eye(num_channels, num_channels)
		kernel = tf.constant(kernel_value)

		# do the un-pooling using conv2d_transpose
		unpool = tf.nn.conv2d_transpose(input_tensor,
										kernel,
										output_shape=self.output_shape,
										strides=(1, kernel_rows, kernel_cols, 1),
										padding='VALID')
		# TODO test!!!
		return unpool

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)


class Unfold(Layer):
	"""

	"""
	def __init__(self,
				 scope=''):
		Layer.__init__(self)

		self.scope = scope

	def build(self, input_tensor):
		num_batch, height, width, num_channels = input_tensor.get_shape()

		return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)

	def decoder(self, output_shape, activation):
		return Fold(output_shape, scope=self.scope+"_decoder")

	def get_params(self):
		return []

	def get_param_names(self):
		return []


class Fold(Layer):
	"""

	"""
	def __init__(self,
				 fold_shape,
				 scope=''):
		Layer.__init__(self)

		self.fold_shape = fold_shape
		self.scope = scope

	def build(self, input_tensor):
		return tf.reshape(input_tensor, self.fold_shape)

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)


class FullyConnected(Layer):
	"""

	"""
	def __init__(self,
				 output_dim,
				 weights=None,
				 bias=None,
				 activation=None,
				 scope=''):
		Layer.__init__(self)

		self.output_dim = output_dim
		self.weights = weights
		self.bias = bias
		self.activation = activation
		self.scope = scope

	def build(self, input_tensor):
		num_batch, input_dim = input_tensor.get_shape()

		# build weights
		if self.weights is not None:
			print(self.weights.get_shape())
			print(input_dim.value)
			print(self.output_dim)
			assert self.weights.get_shape() == (input_dim.value, self.output_dim)
		else:
			self.weights = tf.Variable(tf.truncated_normal((input_dim.value, self.output_dim), stddev=0.1),
									   name='weights')
			print("encode: ", self.weights.get_shape())

		# build bias
		if self.bias:
			assert self.bias.get_shape() == (self.output_dim, )
		else:
			self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_dim]), name='bias')

		# fully connected layer
		fc = tf.matmul(input_tensor, self.weights) + self.bias

		# activation
		if self.activation:
			return self.activation(fc)
		return fc

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)

	def decoder(self, output_shape, activation):
		output_dim, input_dim = self.weights.get_shape()
		return FullyConnected(output_dim, weights=tf.transpose(self.weights), activation=activation, scope=self.scope+"_decoder")

	def get_params(self):
		return [self.weights, self.bias]

	def get_param_names(self):
		return [self.scope+"_W", self.scope+"_b"]


class Corrupt(Layer):
	"""

	"""
	def __init__(self,
				 noise_stdev=0.1,
				 scope=''):
		Layer.__init__(self)

		self.noise_stdev = noise_stdev
		self.scope = scope

	def build(self, input_tensor):
		noise = tf.random_normal(shape=tf.shape(input_tensor), mean=0.0, stddev=self.noise_stdev, dtype=tf.float32)
		corrupt = tf.minimum(tf.maximum(input_tensor+noise, 0.0), 1.0)
		return corrupt

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build(input_tensor)
		else:
			return self.build(input_tensor)

	def decoder(self, output_shape, activation):
		return None

	def get_params(self):
		return []

	def get_param_names(self):
		return []

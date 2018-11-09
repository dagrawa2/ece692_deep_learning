import gpflow
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

from gpflow.params import Parameterized
from gpflow.params import Parameter
from gpflow.decors import name_scope
from gpflow.decors import params_as_tensors
from gpflow import settings

from . import utils
from .decors import name_scope_for_layer


class Layer(Parameterized, metaclass=ABCMeta):

	def __init__(self, name=None):
		super().__init__(name=name)

	@abstractmethod
	def build_layer(self, *args, **kwargs):
		raise NotImplementedError

	def __call__(self, *args, **kwargs):
		return self.build_layer(*args, **kwargs)


class Convolution2D(Layer):

	def __init__(self, kernel_shape, strides=(1, 1, 1, 1), padding='SAME', activation=None, name=''):
		Layer.__init__(self)
		self.kernel_shape = kernel_shape
		self.strides = strides
		self.padding = padding
		self.activation = activation
		self._name = name
		self.kernel = Parameter(utils.truncated_normal(scale=0.1, size=self.kernel_shape, dtype=settings.float_type), name='kernel')
		self.bias = Parameter(0.1*np.ones((kernel_shape[-1]), dtype=settings.float_type), name='bias')

	@name_scope_for_layer
	@params_as_tensors
	def build_layer(self, input_tensor):
		conv = tf.nn.conv2d(input_tensor, self.kernel, strides=self.strides, padding=self.padding)
		if self.activation:
			return self.activation(conv + self.bias)
		return conv + self.bias

	def get_params(self):
		return [self.kernel, self.bias]


class MaxPooling(Layer):

	def __init__(self, kernel_shape=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name=''):
		Layer.__init__(self)
		self.kernel_shape = kernel_shape
		self.strides = strides
		self.padding = padding
		self._name = name

	@name_scope_for_layer
	@params_as_tensors
	def build_layer(self, input_tensor):
		return tf.nn.max_pool(input_tensor, ksize=self.kernel_shape, strides=self.strides, padding=self.padding)

	def call(self, input_tensor):
		if self.scope:
			with tf.variable_scope(self.scope) as scope:
				return self.build_layer(input_tensor)
		else:
			return self.build_layer(input_tensor)

	def get_params(self):
		return []


class Unfold(Layer):

	def __init__(self, name=''):
		Layer.__init__(self)
		self._name = name

	@name_scope_for_layer
	@params_as_tensors
	def build_layer(self, input_tensor):
		num_batch, height, width, num_channels = input_tensor.get_shape()
		return tf.reshape(input_tensor, [-1, (height * width * num_channels).value])

	def get_params(self):
		return []


class FullyConnected(Layer):

	def __init__(self, dims, activation=None, name='fc'):
		Layer.__init__(self)
		self.dims = dims
		self.activation = activation
		self._name = name
		self.weights = Parameter(utils.truncated_normal(scale=0.1, size=tuple(self.dims), dtype=settings.float_type), name='weights')
		self.bias = Parameter(0.1*np.ones((self.dims[1]), dtype=settings.float_type), name='bias')

	@name_scope_for_layer
	@params_as_tensors
	def build_layer(self, input_tensor):
		fc = tf.matmul(input_tensor, self.weights) + self.bias
		if self.activation:
			return self.activation(fc)
		return fc

	def get_params(self):
		return [self.weights, self.bias]


class Input(Layer):

	def __init__(self, shape, name=''):
		Layer.__init__(self)
		self.shape = shape
		self._name = name
		self.dim = np.prod(np.array(shape))

	@name_scope_for_layer
	@params_as_tensors
	def build_layer(self, input_tensor):
		return tf.reshape(input_tensor, [-1]+self.shape)

	def get_params(self):
		return []

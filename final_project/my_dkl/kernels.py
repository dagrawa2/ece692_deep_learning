import gpflow
import tensorflow as tf
import numpy as np
from .layers import *

from gpflow.kernels import Kernel
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter


class NNGP_original(Kernel):

	def __init__(self, input_dim, active_dims=None, name=None, v_w=1.0, v_b=1.0, depth=1):
		super().__init__(input_dim, active_dims, name=name)
		self.v_b = Parameter(v_b, transform=gpflow.transforms.positive)
		self.v_w = Parameter(v_w, transform=gpflow.transforms.positive)
		self.depth = depth

	@params_as_tensors
	def K(self, X, X2=None, presliced=False):
		if not presliced:
			X, X2 = self._slice(X, X2)
		if X2 is None:
			K_XX = self.v_b + self.v_w*tf.matmul(X, X, transpose_b=True)
			K_X = self.v_b + self.v_w*tf.reshape(tf.reduce_sum(X**2, axis=1), [-1, 1])
			for d in range(self.depth):
				N = tf.sqrt(tf.matmul(K_X, K_X, transpose_b=True))
				R = tf.maximum(tf.minimum(K_XX/N, 1), -1)
				K_XX = self.v_b + self.v_w*N/2*tf.where(tf.abs(R)<1, (tf.sqrt(1-R**2)+(np.pi-tf.acos(R))*R)/np.pi, (1+R)/2)
				K_X = self.v_b + self.v_w*K_X/2
			return K_XX
		else:
			K_XX2 = self.v_b + self.v_w*tf.matmul(X, X2, transpose_b=True)
			K_X = self.v_b + self.v_w*tf.reshape(tf.reduce_sum(X**2, axis=1), [-1, 1])
			K_X2 = self.v_b + self.v_w*tf.reshape(tf.reduce_sum(X2**2, axis=1), [-1, 1])
			for d in range(self.depth):
				N = tf.sqrt(tf.matmul(K_X, K_X2, transpose_b=True))
				R = tf.maximum(tf.minimum(K_XX2/N, 1), -1)
				K_XX2 = self.v_b + self.v_w*N/2*tf.where(tf.abs(R)<1, (tf.sqrt(1-R**2)+(np.pi-tf.acos(R))*R)/np.pi, (1+R)/2)
				K_X = self.v_b + self.v_w*K_X/2
				K_X2 = self.v_b + self.v_w*K_X2/2
			return K_XX2

	@params_as_tensors
	def Kdiag(self, X, presliced=False):
		if not presliced:
			X, _ = self._slice(X, None)
		K_X = self.v_b + self.v_w*tf.reduce_sum(X**2, axis=1)
		for d in range(self.depth):
			K_X = self.v_b + self.v_w*K_X/2
		return K_X


class deep_kernel(Kernel):

	def __init__(self, base_kernel, layers, name=None):
		super().__init__(layers[0].dim, active_dims=None, name=name)
		self.base_kernel = base_kernel
		self.layers = []
		for i, layer in enumerate(layers):
			setattr(self, "layer_"+str(i), layer)
			self.layers.append(getattr(self, "layer_"+str(i)))

	@params_as_tensors
	def K(self, X, X2=None, presliced=False):
		if not presliced:
			X, X2 = self._slice(X, X2)
		if X2 is None:
			Z = tf.identity(X)
			for layer in self.layers:
				Z = layer(Z)
			return self.base_kernel.K(Z)
		else:
			Z = tf.identity(X)
			Z2 = tf.identity(X2)
			for layer in self.layers:
				Z = layer(Z)
				Z2 = layer(Z2)
			return self.base_kernel.K(Z, Z2)

	@params_as_tensors
	def Kdiag(self, X, presliced=False):
		if not presliced:
			X, _ = self._slice(X, None)
		Z = tf.identity(X)
		for layer in self.layers:
			Z = layer(Z)
		return self.base_kernel.Kdiag(Z)


class NNGP(gpflow.kernels.Kernel):
	"""
	Kernel of a fully connected NNGP.

	This class implements a GPFlow kernel that results from a deep NNGP architecture with relu activation.
	"""

	def __init__(self, input_dim, active_dims=None, name=None, v_w=1.0, v_b=1.0, depth=1):
		"""
		Initializes the NNGP kernel.

		Parameters
		----------
		input_dim : int
			Dimension of the input space.
		active_dims : None, optional
			This argument is necessary for a GPFlow kernel. Only use the default value for now.
		name : None, optional
			This argument is necessary for a GPFlow kernel. Only use the default value for now.
		v_w : positive float
			Variance of the weights prior. The default value is 1.0.
		v_b positive float
			Variance of the biases prior. The default value is 1.0.
		depth : non-negative int
			Number of hidden layers. The default value is 1.
		"""
		super().__init__(input_dim, active_dims, name=name)
		self.v_b = gpflow.params.Parameter(v_b, transform=gpflow.transforms.positive)
		self.v_w = gpflow.params.Parameter(v_w, transform=gpflow.transforms.positive)
		self.depth = depth

	@gpflow.decors.params_as_tensors
	def K(self, X, X2=None, presliced=False):
		"""
		Computes the kernel matrix given input data matrices.

		For details, see the documentation for gpflow.kernels.Kernel.

		Parameters
		----------
		X : numpy.ndarray
			Firsst input data matrix (e.g., training or test set). It must be 2D with each row being one datum.
		X2 : numpy.ndarray, optional
			Second input data matrix (e.g., training or test set). It must be 2D with each row being one datum. By default, X2 is set equal to X.
		presliced : None, optional
			This argument is necessary for a GPFlow kernel. Only use the default value for now.

		Returns
		-------
		numpy.ndarray
			The 2D kernel matrix for the given data.
		"""
		if not presliced:
			X, X2 = self._slice(X, X2)
		if X2 is None:
			X2 = X
		KXX2 = self.v_b + self.v_w*tf.matmul(X, X2, transpose_b=True)
		KXX = self.v_b + self.v_w*tf.reshape(tf.reduce_sum(X**2, axis=1), [-1, 1])
		KX2X2 = self.v_b + self.v_w*tf.reshape(tf.reduce_sum(X2**2, axis=1), [-1, 1])
		c = 1-1e-9
		for d in range(self.depth):
			norm = tf.sqrt(tf.matmul(KXX, KX2X2, transpose_b=True))
			corr = KXX2/norm
			corr = tf.where(corr<=-1, -c*tf.ones_like(corr), tf.where(corr>=1, c*tf.ones_like(corr), corr))
			KXX2 = self.v_b + self.v_w*norm/2.0*tf.where(tf.abs(corr)<c, (tf.sqrt(1-corr**2)+(np.pi-tf.acos(corr))*corr)/np.pi, tf.where(corr<-1, tf.zeros_like(corr), tf.ones_like(corr)))
			KXX = self.v_b + self.v_w*KXX/2.0
			KX2X2 = self.v_b + self.v_w*KX2X2/2.0
		return KXX2

	@gpflow.decors.params_as_tensors
	def Kdiag(self, X, presliced=False):
		"""
		Computes the diagonal of the kernel matrix K(X, X).

		For details, see the documentation for gpflow.kernels.Kernel.

		Parameters
		----------
		X : numpy.ndarray
			Input data matrix (e.g., training or test set). It must be 2D with each row being one datum.
		presliced : None, optional
			This argument is necessary for a GPFlow kernel. Only use the default value for now.

		Returns
		-------
		numpy.ndarray
			The 1D diagonal of the kernel matrix for the given data.
		"""
		if not presliced:
			X, _ = self._slice(X, None)
		KXX = self.v_b + self.v_w*tf.reduce_sum(X**2, axis=1)
		for d in range(self.depth):
			KXX = self.v_b + self.v_w*KXX/2.0
		return KXX

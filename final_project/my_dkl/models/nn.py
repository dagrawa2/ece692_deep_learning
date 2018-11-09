import gpflow
import tensorflow as tf
import numpy as np

from gpflow import settings

from gpflow.conditionals import base_conditional
from gpflow.params import DataHolder
from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope
from gpflow.logdensities import multivariate_normal

from gpflow.models.model import GPModel

from ..dataholders import Minibatch
from ..decors import use_minibatches, not_use_minibatches
from .. import likelihoods


class NN(GPModel):

	def __init__(self, X, Y, layers, minibatch_size=None, seed=0, **kwargs):
		likelihood = likelihoods.delta()
		self._seed = seed
		self._minibatch_size = minibatch_size
		self._N = X.shape[0]
		if len(X.shape) >= 3: X = X.reshape((-1, np.prod(np.array(list(X.shape)[1:]))))
		if minibatch_size is None:
			X = DataHolder(X)
			Y = DataHolder(Y)
			self._iters_per_epoch = 1
		else:
			X = Minibatch(X, batch_size=self._N, shuffle=True, seed=seed)
			Y = Minibatch(Y, batch_size=self._N, shuffle=True, seed=seed)
			self._iters_per_epoch = self._N//minibatch_size + min(1, self._N%minibatch_size)
		GPModel.__init__(self, X, Y, None, likelihood, None, **kwargs)
		self.layers = []
		for i, layer in enumerate(layers):
			setattr(self, "layer_"+str(i), layer)
			self.layers.append(getattr(self, "layer_"+str(i)))

	@name_scope('likelihood')
	@params_as_tensors
	def _build_likelihood(self):
		Z = tf.identity(self.X)
		for layer in self.layers:
			Z = layer(Z)
		return -tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=Z)

	@name_scope('predict')
	@params_as_tensors
	def _build_predict(self, Xnew, full_cov=False):
		Z = tf.identity(Xnew)
		for layer in self.layers:
			Z = layer(Z)
		return tf.nn.softmax(Z), tf.constant([0], dtype=tf.float32)

	@not_use_minibatches
	def predict_y(self, Xnew):
		return super(NN, self).predict_y(Xnew)

	@use_minibatches
	def train(self, lr=1e-3, eval_set=None, epochs=1):
		opt = gpflow.train.AdamOptimizer(learning_rate=lr)
		if eval_set is not None:
			self.scores = []
			X_test, Y_test = eval_set
			if len(X_test.shape) >= 3: X_test = X_test.reshape((-1, np.prod(np.array(list(X_test.shape)[1:]))))
			opt.minimize(self, maxiter=self._iters_per_epoch*epochs, step_callback=lambda step: self.evaluate(step, X_test, Y_test))
			return np.array(self.scores)
		else:
			opt.minimize(self, maxiter=self._iters_per_epoch*epochs)
			return None

	def evaluate(self, step, X, Y):
		if (step+1)%self._iters_per_epoch == 0:
			pred, _ = self.predict_y(X)
			acc = np.mean(np.equal(np.argmax(Y, axis=1), np.argmax(pred, axis=1)))
			self.scores.append(acc)
			print("Epoch ", step//self._iters_per_epoch, ": Acc ", np.round(acc, 4))

	def save_params(self, filename):
		np.savez(filename, **self.read_trainables())

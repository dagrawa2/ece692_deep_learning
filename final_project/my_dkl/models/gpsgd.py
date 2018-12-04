import gpflow
import tensorflow as tf
import numpy as np

from gpflow import likelihoods
from gpflow import settings

from gpflow.conditionals import base_conditional
from gpflow.params import DataHolder
from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope
from gpflow.logdensities import multivariate_normal

from gpflow.models.model import GPModel

from ..dataholders import Minibatch
from ..decors import use_minibatches, not_use_minibatches


class GPSGD(GPModel):

	def __init__(self, X, Y, kern, mean_function=None, minibatch_size=None, seed=0, **kwargs):
		likelihood = likelihoods.Gaussian()
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
		GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)

	@name_scope('likelihood')
	@params_as_tensors
	def _build_likelihood(self):
		K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
#		K = tf.Print(K, [tf.sqrt(tf.reduce_mean((K-tf.transpose(K))**2))], message="K: ", summarize=10)
#		K = tf.Print(K, [tf.reduce_min(K), tf.reduce_max(K)], message="K: ", summarize=10)
		L = tf.cholesky(K)
		m = self.mean_function(self.X)
		logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y
		return tf.reduce_sum(logpdf)

	@name_scope('predict')
	@params_as_tensors
	def _build_predict(self, Xnew, full_cov=False):
		y = self.Y - self.mean_function(self.X)
		Kmn = self.kern.K(self.X, Xnew)
		Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
		###
#		Kmm_sigma = tf.Print(Kmm_sigma, [tf.shape(Kmm_sigma)], message="K.shape: ", summarize=10)
#		Kmm_sigma = tf.Print(Kmm_sigma, [tf.sqrt(tf.reduce_mean((Kmm_sigma-tf.transpose(Kmm_sigma))**2))], message="K-K.T MSE: ", summarize=10)
#		Kmm_sigma = tf.Print(Kmm_sigma, [tf.reduce_min(Kmm_sigma), tf.reduce_max(Kmm_sigma)], message="K min max: ", summarize=10)
#		eig, _ = tf.linalg.eigh(Kmm_sigma)
#		Kmm_sigma = tf.Print(Kmm_sigma, [tf.reduce_min(eig), tf.reduce_max(eig)], message="K min max eig: ", summarize=10)
		###
		Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
		f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
		return f_mean + self.mean_function(Xnew), f_var

	@not_use_minibatches
	def predict_y(self, Xnew):
		if len(Xnew.shape) >= 3: Xnew = Xnew.reshape((-1, np.prod(np.array(list(Xnew.shape)[1:]))))
		return super(GPSGD, self).predict_y(Xnew)

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
			pred_mean, pred_var = self.predict_y(X)
#			mse = np.sum((pred_mean-Y.reshape(pred_mean.shape))**2)/Y.shape[0]
			acc = np.mean(np.equal(np.argmax(Y, axis=1), np.argmax(pred_mean, axis=1)))
			self.scores.append(acc)
			print("Epoch ", step//self._iters_per_epoch, ": Acc ", np.round(acc, 4))

	def load_params_NN(self, filename):
		params = np.load(filename)
		params = {name: params["NN"+name[10:]] for name in self.read_trainables().keys() if name[11:16] == "layer"}
		self.assign(params)

	def save_params(self, filename):
		np.savez(filename, **self.read_trainables())

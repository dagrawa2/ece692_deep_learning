import tensorflow as tf
from gpflow.decors import params_as_tensors
from gpflow.likelihoods import Likelihood

class delta(Likelihood):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	@params_as_tensors
	def predict_mean_and_var(self, Fmu, Fvar):
		return Fmu, Fvar

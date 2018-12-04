import tensorflow as tf
import scipy.stats as stats

def truncated_normal(scale=1, size=(1), dtype=None):
	A = stats.truncnorm.rvs(-2, 2, scale=scale, size=size)
	if dtype is not None: A = A.astype(dtype)
	return A

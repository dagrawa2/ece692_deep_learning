import tensorflow as tf
import functools

def use_minibatches(method):
	@functools.wraps(method)
	def wrapper(obj, *args, **kwargs):
		orig_batch_size = obj.X.batch_size
		obj.X.batch_size = obj._minibatch_size
		obj.Y.batch_size = obj._minibatch_size
		result = method(obj, *args, **kwargs)
		obj.X.batch_size = orig_batch_size
		obj.Y.batch_size = orig_batch_size
		return result
	return wrapper

def not_use_minibatches(method):
	@functools.wraps(method)
	def wrapper(obj, *args, **kwargs):
		orig_batch_size = obj.X.batch_size
		obj.X.batch_size = obj._N
		obj.Y.batch_size = obj._N
		result = method(obj, *args, **kwargs)
		obj.X.batch_size = orig_batch_size
		obj.Y.batch_size = orig_batch_size
		return result
	return wrapper


def name_scope_for_layer(method):
	@functools.wraps(method)
	def wrapper(obj, *args, **kwargs):
		scope_name = obj._name if obj._name is not None else method.__name__
		with tf.name_scope(scope_name):
			return method(obj, *args, **kwargs)
	return wrapper



import numpy as np
import time
from copy import deepcopy

def linear(X):
	return X

def logistic(X):
	return 1/(1+np.exp(-X))

def logistic_deriv(Y):
	return Y*(1-Y)

def relu(X):
	return np.maximum(0, X)

def relu_deriv(Y):
		return np.heaviside(Y, 0)

def softmax(X):
	temp = np.exp(X)
	return temp/np.sum(temp, axis=1, keepdims=True)

def cross_entropy(Y, Z):
	return -1/Y.shape[0]*np.sum(Y*np.where(Y>0, np.log(Z), Z)+(1-Y)*np.where(Y<1, np.log(1-Z), Z))

def entropy(Y, Z):
	return -np.mean(np.sum(Y*np.where(Y>0, np.log(Z), Z), axis=1))

def square_error(Y, Z):
	return np.mean(np.sum((Z-Y)**2, axis=1)/2)

def accuracy(Y, Z):
	return np.mean(np.argmax(Y, axis=1)==np.argmax(Z, axis=1))


class NN:

	def __init__(self, layers, activations, learning_rate=0.1, mini_batch_size=1, lambda_1=0, lambda_2=0, random_state=None, force_square_error=False):
		self.layers = layers
		self.activations = activations
		self.activation_derivs = []
		for act in activations[:-1]:
			if act == logistic: self.activation_derivs.append(logistic_deriv)
			elif act == relu: self.activation_derivs.append(relu_deriv)
		self.activation_derivs.append(None)
		if self.activations[-1] == logistic: self.loss = cross_entropy
		elif self.activations[-1] == softmax: self.loss = entropy
		elif self.activations[-1] == linear: self.loss = square_error
		self.force_square_error = force_square_error
		if force_square_error: self.loss = square_error
		self.eta = learning_rate
		self.mbs = mini_batch_size
		self.lambda_1 = lambda_1
		self.lambda_2 = lambda_2
		np.random.seed(random_state)
		self.W = [np.random.normal(0, 1/np.sqrt(m), (m, n)) for m,n in zip(layers[:-1], layers[1:])]
		self.b = [np.random.normal(0, 1, (1, n)) for n in layers[1:]]

	def predict(self, X):
		A = np.atleast_2d(deepcopy(X))
		for act,W,b in zip(self.activations, self.W, self.b):
			A = act(A.dot(W)+b)
		return A

	def forward(self, X):
		a = [np.atleast_2d(deepcopy(X))]
		for act,W,b in zip(self.activations, self.W, self.b):
			a.append( act(a[-1].dot(W)+b) )
		return a

	def backward(self, Y, a):
		deltas = [(a[-1]-Y)*a[1]*(1-a[-1])] if self.force_square_error else [a[-1]-Y]
		for W,act_deriv,a_ in list(zip(self.W[1:], self.activation_derivs[:-1], a[1:-1]))[::-1]:
			deltas.append( deltas[-1].dot(W.T)*act_deriv(a_) )
		deltas.reverse()
		for W,b,a_,delta in zip(self.W, self.b, a[:-1], deltas):
#			"""
			i = np.random.randint(0, W.shape[0])
			j = np.random.randint(0, W.shape[1])
			grad = 1/Y.shape[0]*a_.T.dot(delta)
			self.grads.append(grad[i, j])
			W[i, j] += self.eps
			L_1 = self.loss(Y, self.predict(a[0]))
			W[i, j] -= 2*self.eps
			L_2 = self.loss(Y, self.predict(a[0]))
			self.num_grads.append((L_1-L_2)/(2*self.eps))
			W[i, j] += self.eps
#			"""
#			i = np.random.randint(0, b.shape[1])
#			grad = 1/Y.shape[0]*np.sum(delta, axis=0, keepdims=True)
#			self.grads.append(grad[0,i])
#			b[0,i] += self.eps
#			L_1 = self.loss(Y, self.predict(a[0]))
#			b[0,i] -= self.eps
#			L_2 = self.loss(Y, self.predict(a[0]))
#			self.num_grads.append((L_1-L_2)/(2*self.eps))
#			b[0,i] += self.eps
			W -= self.eta/Y.shape[0]*a_.T.dot(delta) + self.eta/self.n_train*(self.lambda_1*np.sign(W)+self.lambda_2*W)
			b -= self.eta/Y.shape[0]*np.sum(delta, axis=0, keepdims=True)

	def train(self, X_train, Y_train, eval_set=None, epochs=1, early_stopping=None):
		self.eps = 0.0001
		self.grads = []
		self.num_grads = []
		X_test, Y_test = eval_set
		self.n_train = X_train.shape[0]
		indices = np.arange(X_train.shape[0])
		accs_test = []
		if early_stopping is None: early_stopping = epochs+1
		for epoch in range(epochs):
#			print("Epoch ", epoch, " . . . ")
#			np.random.shuffle(indices)
			X = X_train[indices]
			Y = Y_train[indices]
			for X_batch,Y_batch in [(X[i:min(self.n_train,i+self.mbs)], Y[i:min(self.n_train,i+self.mbs)]) for i in range(0,self.n_train,self.mbs)]:
				self.backward(Y_batch, self.forward(X_batch))
			accs_test.append(accuracy(Y_test, self.predict(X_test)))
#			print("\t Test accuracy: ", np.round(accs_test[-1], 6))
			if 1+epoch >= early_stopping and np.argmax(accs_test[-early_stopping:])==0: break
		return np.array(accs_test)

import time
import tensorflow as tf
import numpy as np

class autoencoder:

	def __init__(self, encoder_layers, blocks, lr=0.01, lr_decay=1, mbs=1, pred_mbs=None, seed=None):
		self.encoder_layers = encoder_layers
		self.blocks = blocks
		self.lr = lr
		self.lr_decay = lr_decay
		self.mbs = mbs
		self.pred_mbs = pred_mbs
		if seed is not None: tf.set_random_seed(seed)
		self.x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
		self.build_autoencoder()

	def build_encoder(self):
		self.out_shapes = []
		self.params = []
		self.param_names = []
		out = tf.identity(self.x)
		for layer in self.encoder_layers:
			self.out_shapes.append(tf.shape(out))
			out = layer(out)
			self.params = self.params + layer.get_params()
			self.param_names = self.param_names + layer.get_param_names()
		self.encoded = tf.identity(out)

	def build_decoder_layers(self):
		first_param_layer = -1
		for layer in self.encoder_layers:
			first_param_layer += 1
			if len(layer.get_param_names()) > 0:
				break
		self.decoder_layers = []
		for i, (layer, output_shape) in enumerate(list(zip(self.encoder_layers, self.out_shapes))[::-1]):
			activation = tf.nn.sigmoid if i == first_param_layer else tf.nn.relu
			self.decoder_layers.append( layer.decoder(output_shape, activation) )

	def build_decoder(self):
		out = tf.identity(self.encoded)
		for layer in self.decoder_layers:
			if layer is not None:
				out = layer(out)
		self.reconstruction = tf.identity(out)

	def build_block_autoencoder(self, block):
		self.block_input = tf.placeholder(tf.float32, shape=self.out_shapes[block[0]])
		out = tf.identity(self.block_input)
		for layer in [self.encoder_layers[i] for i in block]:
			out = layer(out)
		self.block_encoded = tf.identity(out)
		for layer in [self.decoder_layers[-1-i] for i in block]:
			out = layer(out)
		loss = tf.nn.l2_loss(self.block_input-out)
		self.pretrain_step = tf.train.AdamOptimizer(self.lr).minimize(loss)

	def build_autoencoder(self):
		self.build_encoder()
		self.build_decoder_layers()
		self.build_decoder()
		self.loss = tf.nn.l2_loss(self.x - self.reconstruction)
		self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def start_session(self):
		self.sess = tf.Session()
		init = tf.global_variables_initializer()
		self.sess.run(init)

	def stop_session(self):
		self.sess.close()

	def pretrain(self, X_train, epochs=1):
		n_train = X_train.shape[0]
		n_batches = int(np.ceil(n_train/self.mbs))
		mb_progress = int(0.1*n_batches)
		X = np.copy(X_train)
		for block in self.blocks:
			print("Block ", block)
			self.build_block_autoencoder(block)
			for epoch in range(epochs):
				print("Epoch ", epoch, " . . . ")
				time_0 = time.time()
				np.random.shuffle(X)
				time_1 = time.time()
				for k, i in enumerate(range(0,n_train,self.mbs)):
#					if k%mb_progress == 0:
#						print(k, "/", n_batches, " batches (", np.round(time.time()-#time_1, 5), " s)")
#						time_1 = time.time()
					X_batch = X[i:min(n_train,i+self.mbs)]
					self.sess.run([self.pretrain_step], feed_dict={self.block_input: X_batch})
			X = self.block_encode(X)
		return

	def block_encode(self, X):
		n_test = X.shape[0]
		batches = []
		for i in range(0,n_test,self.pred_mbs):
			X_batch = X[i:min(n_test,i+self.pred_mbs)]
			batches.append( self.sess.run(self.block_encoded, feed_dict={self.block_input: X_batch}) )
		return np.concatenate(tuple(batches), axis=0)


	def train(self, X_train, X_test, epochs=1, early_stopping=None):
		if self.pred_mbs is None: self.pred_mbs = X_test.shape[0]
		n_train = X_train.shape[0]
		n_batches = int(np.ceil(n_train/self.mbs))
		mb_progress = int(0.1*n_batches)
		indices = np.arange(X_train.shape[0])
		losses = []
		best_parameters = self.sess.run(self.params)
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
				best_parameters = self.sess.run(self.params)
			if 1+epoch >= early_stopping and np.argmin(losses[-early_stopping:])==0: break
		best_parameters = {name: value for name, value in zip(self.param_names, best_parameters)}
		return np.array(losses), best_parameters

	def test_loss(self, X):
		n_test = X.shape[0]
		batch_losses = []
		for i in range(0,n_test,self.pred_mbs):
			X_batch = X[i:min(n_test,i+self.pred_mbs)]
			batch_losses.append( self.sess.run(self.loss, feed_dict={self.x: X_batch}) )
		return np.sum(batch_losses)/n_test

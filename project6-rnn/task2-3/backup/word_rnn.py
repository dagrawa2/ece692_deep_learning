import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell,  GRUCell
import sys

class character_rnn(object):
	'''
	sample character-level RNN by Shang Gao
	
	parameters:
	  - seq_len: integer (default: 200)
		number of characters in input sequence
	  - first_read: integer (default: 50)
		number of characters to first read before attempting to predict next character
	  - rnn_size: integer (default: 200)
		number of rnn cells
	   
	methods:
	  - train(text, iterations=100000)
		train network on given text
	'''
	def __init__(self, seq_len=50, first_read=4, rnn_size=200, vocab=None, embeddings=None):
	
		self.seq_len = seq_len
		self.first_read = first_read
	
		#dictionary of possible characters
		self.words = vocab
		self.num_words = len(self.words)
		
		'''
		#training portion of language model
		'''

		#input sequence of word indices
		self.input = tf.placeholder(tf.int32, [1, seq_len])
		
		#convert to one_hot and embeddings
		one_hot = tf.one_hot(self.input, self.num_words)
		ems = tf.reshape(tf.gather(tf.constant(embeddings), tf.reshape(self.input, [seq_len])), [1, seq_len, embeddings.shape[1]])
		
		#rnn layer
		self.gru = GRUCell(rnn_size)
		outputs, states = tf.nn.dynamic_rnn(self.gru, ems, sequence_length=[seq_len], dtype=tf.float32)
		outputs = tf.squeeze(outputs, [0])

		#ignore all outputs during first read steps
		outputs = outputs[first_read:-1]
		
		#softmax logit to predict next word (actual softmax is applied in cross entropy function)
#		dual_embeddings = embeddings.dot(np.linalg.pinv(embeddings.T.dot(embeddings)))
#		logits = tf.matmul(tf.layers.dense(outputs, embeddings.shape[1], None, True, tf.orthogonal_initializer(), name='dense'), dual_embeddings, transpose_b=True)
		logits = tf.layers.dense(outputs, embeddings.shape[1], None, True, tf.orthogonal_initializer(), name='dense')
		logits = -tf.reduce_sum((tf.expand_dims(logits, -1) - tf.constant(np.expand_dims(embeddings.T, 0)))**2, 1)

		#target word at each step (after first read chars) is following character		
		targets = one_hot[0, first_read+1:]
		
		#loss and train functions
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
		self.optimizer = tf.train.AdamOptimizer(0.0002, 0.9, 0.999).minimize(self.loss)
		
		'''
		#generation portion of language model
		'''
		
		#use output and state from last word in training sequence
		state = tf.expand_dims(states[-1], 0)
		output = ems[:, -1, :]
		
		#save predicted words to list
		self.predictions = []
		
		#generate 25 new words that come after input sequence
		for i in range(25):
		
			#run GRU cell and softmax 
			output, state = self.gru(output, state)
#			logits = tf.matmul(tf.layers.dense(output, embeddings.shape[1], None, True, tf.orthogonal_initializer(), name='dense', reuse=True), dual_embeddings, transpose_b=True)
			logits = tf.layers.dense(output, embeddings.shape[1], None, True, tf.orthogonal_initializer(), name='dense', reuse=True)
			logits = -tf.reduce_sum((tf.expand_dims(logits, -1) - tf.constant(np.expand_dims(embeddings.T, 0)))**2, 1)

			#get index of most probable word
			output = tf.argmax(logits, 1)

			#save predicted word to list
			self.predictions.append(output)
			
			#embed
			output = tf.gather(tf.constant(embeddings), output)
		
		#init op
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def train(self, text_indices, iterations=100000, fp_loss=None, fp_sample=None):
		'''
		train network on given text
				
		parameters:
		  - text_indices: list of integers
			word indices to train network on
		  - iterations: int (default: 100000)
			number of iterations to train for
		
		outputs:
			None
		'''

		fp_loss.write("iteration,loss\n")
		fp_sample.write("iteration,sample\n")
	
		#get length of text
		text_len = len(text_indices)
		
		#train
		for i in range(iterations):
		
			#select random starting point in text
			start = np.random.randint(text_len - self.seq_len)
			sequence = text_indices[start:start+self.seq_len]
			
			#train
			feed_dict = {self.input:[sequence]}
			loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
#			sys.stdout.write("iterations %i loss: %f  \r" % (i+1, loss))
#			sys.stdout.flush()
			fp_loss.write(str(i+1)+","+str(loss)+"\n")
		
			#show generated sample every 100 iterations
			if (i+1) % 100 == 0:
			
				feed_dict = {self.input:[sequence]}
				pred = self.sess.run(self.predictions, feed_dict=feed_dict)
				sample = ''.join([self.words[idx[0]]+" " for idx in pred])
#				print "iteration %i generated sample: %s" % (i+1, sample)
				fp_sample.write(str(i+1)+","+sample+"\n")

		
if __name__ == "__main__":

	import re
	import string

	#load sample text
	with open('hobbit.txt', 'r') as f:
		text = f.read()
		
	#clean up text
	text = re.sub(r'-|\t|\n', ' ', text)
	text = text.translate(None, string.punctuation).lower()
	text = re.sub(' +', ' ', text) #remove duplicate spaces

	embeddings = np.load("embeddings.npy")
	with open("vocab.txt", "r") as fp:
		vocab = [word.strip("\n") for word in fp.readlines()]

	word2idx = {word: i for i, word in enumerate(vocab)}
	text = [word2idx[w] for w in text.split(" ") if w in vocab]
	print "text word count:", len(text)
	
	#train rnn
	rnn = character_rnn(vocab=vocab, embeddings=embeddings)
	with open("losses.csv", "w") as fp_loss, open("samples.csv", "w") as fp_sample:
		rnn.train(text, fp_loss=fp_loss, fp_sample=fp_sample)

	
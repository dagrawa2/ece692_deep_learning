#sample code implementing CBOW Word2Vec by Shang Gao

import numpy as np
import unicodedata
import gensim
import string
import re
import collections
import logging
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',  level=logging.INFO)

#load corpus
print "loading dataset"
with open("hobbit.txt",  "rb") as fp:
	dataset = fp.read().decode("UTF-8")
dataset = unicodedata.normalize('NFKD',  dataset).encode('ascii', 'ignore')

#preprocess dataset
print "Preprocessing dataset"
dataset = re.sub(r"Chapter(.+?)\n+(.+?)\n", " ", dataset)
dataset = re.sub(r"-|\t|\n|\"", " ", dataset)
dataset = re.sub(r" +", " ", dataset)
dataset = re.sub(r"\?|!|;|:", ".", dataset)
dataset = re.sub(r"\. ([a-z])", " \g<1>", dataset)
dataset = dataset.lower()

#save preprocessed text
print "Saving preprocessed text"
with open("text.txt", "w") as fp:
	fp.write( re.sub(r" +", " ", dataset.translate(None, string.punctuation)) )

#convert dataset to list of sentences
print "converting dataset to list of sentences"
sentences = dataset.split('.')
sentences = [sentence[1:].translate(None,  string.punctuation).lower().split() for sentence in sentences]

#train word2vec
print "training word2vec"
model = gensim.models.Word2Vec(sentences,  min_count=5,  size=200,  workers=4,  iter=20)

#save vocabulary
print "Saving vocabulary"
vocab_size = len(model.wv.vocab)
print "vocab_size = "+str(vocab_size)
vocab = [model.wv.index2word[i] for i in range(vocab_size)]
with open("vocab.txt",  "w") as fp:
	for word in vocab[:-1]:
		fp.write(word+"\n")
	fp.write(vocab[-1])

#save embeddings
print "Saving embeddings"
embeddings = np.stack([model.wv[word] for word in vocab],  axis=0)
np.save("embeddings.npy",  embeddings)

#get most common words
print "getting common words"
dataset = [item for sublist in sentences for item in sublist]
counts = collections.Counter(dataset).most_common(500)

#reduce embeddings to 2d using tsne
print "reducing embeddings to 2D"
embeddings = np.empty((500, 200))
for i in range(500):
	embeddings[i, :] = model[counts[i][0]]
tsne = TSNE(perplexity=30,  n_components=2,  init='pca',  n_iter=7500)
embeddings = tsne.fit_transform(embeddings)

#plot embeddings
print "plotting most common words"
fig,  ax = plt.subplots(figsize=(30,  30))
for i in range(500):
	ax.scatter(embeddings[i, 0], embeddings[i, 1])
	ax.annotate(counts[i][0],  (embeddings[i, 0], embeddings[i, 1]))

#save to disk
plt.savefig('plot.png')

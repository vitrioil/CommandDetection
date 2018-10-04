import nltk
import pyemd
import scipy
import pickle
import sklearn
import numpy as np

class WordNotFound(Exception):
	
	def __init__(self, message: str, error: str):
		super().__init__(message)
		self.error = error

	def __str__(self):
		return message 

class WMD:
	def __init__(self, path="../input/word_embedding.npy", vocab_path="../input/vocab_pruned.pickle"):
		self.path = path
		self.vocab_path = vocab_path
		self.embedding, self.word_set = self.load_embedding(path)
		self.rev_set = dict(zip(self.word_set.values(), self.word_set.keys()))
		self.vocab_len, self.embedding_size = self.embedding.shape
		self.stopwords = set(nltk.corpus.stopwords.words("english")) 

	def load_embedding(self) -> (np.ndarray, dict):
		'''
			Load the embedding matrix.
			
			returns : (embedding, word_set) memory mapped word embedding matrix and dictionary of words and their mapping
		'''
		embedding = np.load(self.path, mmap_mode="r")
		with open(self.vocab_path, "rb") as f:
			vocab = pickle.load(f)
		word_set = {word:indx for indx,word in enumerate(vocab)}
	
		return embedding, word_set
	
	def tokenize(self, sent: str) -> list:
		'''
			sent: string of sentence

			returns: list of tokenized words using nltk.word_tokenize method
		'''

		return nltk.word_tokenize(sent)

	def remove_stopwords(self, sent_list: list) -> list:
		'''
			Removes stop words taken from english vocabulary

			sent_list: input tokenized list

			returns: tokenized and without stopwords list 
		'''
		return [i for i in sent_list if i not in self.stopwords]

	def _sent_to_sparse(self, sent: (str, list), remove_stop = True) -> scipy.sparse.csr_matrix:
		'''
			Convert input sentence to bag of words form

			sent: input str/list sentence

			returns: Normalized bag of words sparse matrix 
			raises WordNotFound if word is not in vocabulary
		'''
		if isinstance(sent, str):
			sent = self.tokenize(sent)
		if remove_stop:
			sent = self.remove_stopwords(sent)
		col = []
		for word in sent:
			if self.word_set.get(word) is None:
				raise WordNotFound(f"Word {word} not in vocab")
			indx = self.word_set[word]
			if indx not in col:
				col.append(indx)

		row = [0]*len(col)
		words, count = np.unique(sent, return_counts=True)
		count_map = dict(zip(map(self.word_set.get,words), count))
		data = [count_map[indx] for indx in col]
		
		sparse = scipy.sparse.csr_matrix((data, (row, col)), shape=(1, self.vocab_len), dtype=np.float64)
		sparse = sklearn.preprocessing.normalize(sparse, norm="l1", copy=False)
		return sparse

	def wmd(self, sent1="this is sentence", sent2="this is sentence") -> float:
		'''
			Calculates the word movers distance between two
			sentences.

			sent1,sent2: Two input sentences in text form
		'''
		sp1 = self._sent_to_sparse(sent1)
		sp2 = self._sent_to_sparse(sent2)
		if sp1 is None or sp2 is None:
			return 
		union_idx = np.union1d(sp1.indices, sp2.indices)

		W = sklearn.metrics.euclidean_distances(self.embedding[union_idx])
		W = W.astype("float64")	

		sp1 = sp1[:, union_idx].A.ravel()
		sp2 = sp2[:, union_idx].A.ravel()
		print(sp1,sp2)
		#import sys
		#def hook(f, *_):
		#	print(f.f_code.co_name)
		#sys.setprofile(hook)
		return pyemd.emd(sp1, sp2, W)

	def emd(self, vec1, vec2, d):
		pass

if __name__ == "__main__":
	wmd = WMD("../input/word_embedding.npy")
	while True:
		sent1 = input("Sentence 1: ")
		sent2 = input("Sentence 2: ")
		print("Distance is", wmd.wmd(sent1, sent2))

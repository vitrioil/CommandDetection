import os
import gc
import sys
import time
import nltk
import keras
import gensim
import string
import pickle
import psutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import wordnet as wn

pos_to_wordnet_dict = {
    'JJ'  :wn.ADJ,
    'JJR' :wn.ADJ,
    'JJS' :wn.ADJ,
    'RB'  :wn.ADV,
    'RBR' :wn.ADV,
    'RBS' :wn.ADV,
    'NN'  :wn.NOUN,
    'NNP' :wn.NOUN,
    'NNS' :wn.NOUN,
    'NNPS':wn.NOUN,
    'VB'  :wn.VERB,
    'VBG' :wn.VERB,
    'VBD' :wn.VERB,
    'VBN' :wn.VERB,
    'VBP' :wn.VERB,
    'VBZ' :wn.VERB,
}
punctuations = set(string.punctuation)

def prepare_logger():
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	handler = logging.FileHandler("train.log")
	handler.setLevel(logging.INFO)

	logger.addHandler(handler)
	return logger

logger = prepare_logger()
stopwords = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
def memory_above(critical_val = 95):
	return (psutil.virtual_memory().percent > critical_val)

def imdb_dataset():
	NUM_WORDS=1000 # only use top 1000 words
	INDEX_FROM=3   # word index offset
	
	train,test = keras.datasets.imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
	train_x,train_y = train
	test_x,test_y = test
	
	word_to_id = keras.datasets.imdb.get_word_index()
	word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
	word_to_id["<PAD>"] = 0
	word_to_id["<START>"] = 1
	word_to_id["<UNK>"] = 2
	
	id_to_word = {value:key for key,value in word_to_id.items()}
	document = []
	for sent_id in train_x:
		sentence = []
		for word_id in sent_id:
			sentence.append(id_to_word[word_id])
		document.append(sentence)
	return np.array(document), train_x 

def shuffle(f, length=None):
	if isinstance(f, pd.io.parsers.TextFileReader):
		return f
	if length is None:
		length = len(f)
	if not isinstance(f, np.ndarray):
		f = np.array(f)
	perm = np.random.permutation(length)
	f = f[perm]
	return f

def tokenize(document):
	for line in document:
		yield nltk.word_tokenize(line)

def remove_stopwords(tokenized_list):
	filtered_list = []
	for word in tokenized_list:
		if word not in stopwords and word not in ["<UNK>","<START>","<END>","<PAD>"] and word not in punctuations:
			filtered_list.append(word.lower())
	return filtered_list

def lemmatize_words(tokenized_list):
	filtered_list = []
	word_pos = nltk.pos_tag(tokenized_list)
	for word, pos in word_pos:
		word = lemmatizer.lemmatize(word, pos_to_wordnet_dict.get(pos,'n')) 
		filtered_list.append(word)
	return filtered_list

def preprocess(path="../input/train.csv", document=None, tokenized=False, name="processed"):
	word_set = {}
	if document is None:
		df = pd.read_csv(path)
		document = df["Text"]
	total_sentences = len(document)
	if not tokenized:
		document = tokenize(document)
	processed_doc = []
	i = 0
	for sent in document:
		processed_sent = []
		for word in lemmatize_words(remove_stopwords(sent)):
			processed_sent.append(word)
			if word_set.get(word) is None:
				word_indx = len(word_set) + 1
				word_set[word] = word_indx

		processed_doc.append(processed_sent)
		i += 1
		print(f"Finished {100*i/total_sentences:.2f}",end="\r")

	save = input("Save processed document?: ")
	if save.lower() != "no":
		di = {"Text": processed_doc}
		df=pd.DataFrame(di)
		df.to_csv(f"{name}.csv")
	
	#r_idx = np.random.randint(0, total_sentences)
	#print()
	#print(document[r_idx])
	#print(processed_doc[r_idx])
	return len(word_set), word_set

def generate_pairs(word_distance, batch_size, processed_doc, path=""):
	if not path.endswith("csv") and processed_doc is None:
		return
	if path != "":
		processed_doc = pd.read_csv(path, chunksize=batch_size)

	context,target = [],[]
	for chunk in processed_doc:
		chunk = chunk["Text"]
		for sent_indx, tokenized_list in enumerate(chunk):
			#tokenized_list = tokenized_list["Text"]
			if isinstance(tokenized_list, str):
				tokenized_list = eval(tokenized_list)
			for indx, target_word in enumerate(tokenized_list):
				#Sampling is suggested
				context_row = [] 
				for distance in range(indx - word_distance, indx + word_distance):
					if distance == indx or distance < 0 or distance >= len(tokenized_list):
						context_row.append(0)
						continue
					context_row.append(word_set[tokenized_list[distance]])
				context.append(context_row)
				target.append(word_set[target_word])
				#Tensorflow gives not a matrix error for only one example ???
		if len(context) != 1:
			yield np.array(context), np.array(target)[..., np.newaxis]
			context, target = [], []

def initialize_parameters(vocab_size, embedding_size):
	vocab_size += 1 # Adjust for padding
	embeddings = tf.Variable(
			tf.random_uniform(
				[vocab_size, embedding_size],
				-1,
				 1
				)
			)
	
	nce_mat = tf.Variable(
			tf.random_uniform(
				[vocab_size, embedding_size],
				-1,
				 1
			)
		)
	
	nce_bias = tf.Variable(
			tf.zeros(
				[vocab_size]
			)
		)

	return (embeddings, nce_mat, nce_bias)

def model(vocab_size, embedding_size, pair, lr, word_distance):
	embeddings, nce_mat, nce_bias = initialize_parameters(vocab_size, embedding_size)
	(inputs, labels) = pair	

	embed = tf.nn.embedding_lookup(embeddings, inputs)
	sum_embed = tf.reduce_sum(embed, axis=1)
	loss = tf.reduce_mean(
			tf.nn.nce_loss(
				weights = nce_mat,
				biases = nce_bias,
				labels = labels,
				inputs = sum_embed,
				num_sampled = 4,
				num_classes = vocab_size
			)
		)
	optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

	output = sum_embed @ tf.transpose(nce_mat) + nce_bias

	correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(labels, 1)), tf.float32),name="accuracy")
	return optimizer, loss, correct_prediction

def train(document, vocab_size, embedding_size = 100, word_distance = 5, batch_size = 64, lr = 1e-3, epochs = 10, path_to_doc = ""):
	tf.reset_default_graph()
	timestamp = str(int(time.time()))
	logdir = "model/"+timestamp+"/"

	inputs = tf.placeholder(tf.int32, [None, 2*word_distance])
	labels = tf.placeholder(tf.int32, [None, 1])
	
	optimizer, loss, correct_prediction = model(vocab_size, embedding_size, (inputs, labels), lr, word_distance)
	
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		try:
			sess.run(init)
			saver = tf.train.Saver()
			for e in range(epochs):	
				epoch_loss = 0
				epoch_acc = 0
				for indx,pair in enumerate(generate_pairs(word_distance, batch_size, document, path_to_doc)):
					if memory_above(95):
						log_str = f"epoch:{e} indx:{indx}, terminating! high memory usage"
						logger.info(log_str) 				
						print(log_str)
						del pair
						gc.collect()
						break
					context, target = pair
					_, c  = sess.run(
							[optimizer, loss],
							feed_dict = {inputs:context, labels:target}
						)
					epoch_loss += c
					epoch_acc += 0
					if indx % (100) == 0:
						logger.info(f"{timestamp} epoch:{e} indx:{indx}, loss:{epoch_loss} ") 
						print(f"epoch:{e} indx:{indx}, loss:{epoch_loss} ", end="\r")
					if indx % (1_00_000) == 0:
						saver.save(sess, os.path.join(logdir, "model.ckpt"), indx)
				print(f"\nLoss: {epoch_loss}")
			saver.save(sess, os.path.join(logdir, "model.ckpt"))
			print(timestamp)
		except KeyboardInterrupt as e:
			print(str(e))
			saver.save(sess, os.path.join(logdir, "model.ckpt"))
		finally:
			save_dictionary = input("Save the dictionary?")
			if save_dictionary.lower() != "no":
				with open(logdir+"word_set.pickle","wb") as f:
					pickle.dump(word_set, f)
def cosine_distance(v1, v2):
	"""returns the cosine_distance"""
	return np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

def load_file(path, embedding_size):
	with open(path+"word_set.pickle","rb") as f:
		word_set = pickle.load(f)
	
	#Tensor weight
	embeddings, nce_mat, nce_bias = initialize_parameters(len(word_set), embedding_size)
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, path+"model.ckpt")
		embedding = embeddings.eval()
	return word_set,embedding

def test(path, embedding_size):
	word_set, embedding = load_file(path,embedding_size)
	idx_to_word = {idx:word for word,idx in word_set.items()}
	def most_similar(word,topn=5):
		word_indx = word_set.get(word)
		if word_indx is None:
			print("Word not found")
			return None
		vec = embedding[word_indx]
		top_matches = [(None,-np.inf) for i in range(topn)]
		for indx,emb_vec in enumerate(embedding, 1):
			distance = cosine_distance(vec, emb_vec)
			if distance > top_matches[-1][1] and word_indx != indx:
				top_matches[-1] = (idx_to_word[indx], distance)
				top_matches.sort(key = lambda x:x[1], reverse = True)
		print(top_matches)
	return most_similar

if __name__ == "__main__":
	print(sys.argv)
	embedding_size = 150
	if len(sys.argv) >= 2 and sys.argv[1] == "train":
#		document, vocab_size, word_set = preprocess(path="../input/Reviews.csv", name="Reviews_processed")
		with open("word_set.pickle", "rb") as f:
			word_set = pickle.load(f)
		vocab_size = len(word_set)
		train(None, vocab_size, embedding_size=embedding_size, word_distance=4, batch_size=32, lr=1e-3, path_to_doc="Reviews_processed_clean.csv")
	
	if len(sys.argv) >= 3 and sys.argv[1] == "test":
		most_similar = test(f"model/{sys.argv[2]}/",embedding_size)
		while True:
			word = input("=>")
			most_similar(word)

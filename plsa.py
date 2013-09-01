#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
lipiji.sdu@gmail.com
zhizhihu.com
----------------------
Bug exsit.

----------------------
Reference:
[1]https://github.com/hitalex/PLSA
'''

import sys
import os
import glob
import re
import numpy as np
from random import random
from operator import itemgetter

def normalize(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0
    
    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s
       

# stop words
STOP_WORDS_DIC = set()
def load_stop_words(sw_file_path):
    sw_file = open(sw_file_path, "r")
    for word in sw_file:
        word = word.replace("\n", "")
        word = word.replace("\r\n", "")
        STOP_WORDS_DIC.add(word)
    sw_file.close()

# di
class Document(object):        
    PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
    CARRIAGE_RETURNS = ['\n', '\r\n']
    WORD_REGEX = "^[a-z']+$"

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(self.filepath)
        self.lines = []
        self.words = []

    def split(self, STOP_WORDS_DIC):
        self.lines = [line for line in self.file]
        for line in self.lines:
            words = line.split(' ')
            for word in words:
                clean_word = self._clean_word(word)
                if clean_word and (clean_word not in STOP_WORDS_DIC) and (len(clean_word) > 1):
                    self.words.append(clean_word)
        self.file.close()


    def _clean_word(self, word):
        word = word.lower()
        for punc in Document.PUNCTUATION + Document.CARRIAGE_RETURNS:
            word = word.replace(punc, '').strip("'")
        return word if re.match(Document.WORD_REGEX, word) else None

#D
class Corpus(object):
    def __init__(self):
        self.documents = []


    def add_document(self, document):
        self.documents.append(document)

    def build_vocabulary(self):
        discrete_set = set()
        for document in self.documents:
            for word in document.words:
                discrete_set.add(word)
        self.vocabulary = list(discrete_set)
        
		
class Plsa(object):
	def __init__(self, corpus, number_of_topics, max_iter, model_path):
		self.n_d = len(corpus.documents)
		self.n_w = len(corpus.vocabulary)
		self.n_t = number_of_topics
		self.max_iter = max_iter
		self.model_path = model_path
		self.L = 0.0 # log-likelihood
		self.error_L = 0.0001; # error for each iter
		self.corpus = corpus		
		# bag of words
		self.n_w_d = np.zeros([self.n_d, self.n_w], dtype = np.int)
		for di, doc in enumerate(corpus.documents):
			n_w_di = np.zeros([self.n_w], dtype = np.int)
			for word in doc.words:
				if word in corpus.vocabulary:
					word_index = corpus.vocabulary.index(word)
					n_w_di[word_index] = n_w_di[word_index] + 1
			self.n_w_d[di] = n_w_di

		# P(z|w,d)
		self.p_z_dw = np.zeros([self.n_d, self.n_w, self.n_t], dtype = np.float)
		# P(z|d)
		self.p_z_d = np.random.random(size=[self.n_d, self.n_t])
		for di in range(self.n_d):
			normalize(self.p_z_d[di])
		# P(w|z)
		self.p_w_z = np.random.random(size = [self.n_t, self.n_w])
		for zi in range(self.n_t):
			normalize(self.p_w_z[zi])

	def log_likelihood(self):
		L = 0
		for di in range(self.n_d):
			for wi in range(self.n_w):
				sum1 = 0
				for zi in range(self.n_t):
					sum1 = sum1 + self.p_z_d[di, zi] * self.p_w_z[zi, wi]
				L = L + self.n_w_d[di, wi] * np.log(sum1)
		return L


	def print_p_z_d(self):
		filename = self.model_path + "p_z_d.txt"
		f = open(filename, "w")
		for di in range(self.n_d):
			f.write("Doc #" + str(di) +":")
			for zi in range(self.n_t):
				f.write(" "+self.p_z_d[di, zi])
			f.write("\n")
		f.close()
	
	def print_p_w_z(self):
		filename = self.model_path + "p_w_z.txt"
		f = open(filename, "w")
		for zi in range(self.n_t):
			f.write("Topic #" + str(zi) +":")
			for wi in range(self.n_w):
				f.write(" "+self.p_w_z[zi, wi])
			f.write("\n")
		f.close()

	def print_top_words(self, topk):
		filename = self.model_path + "top_words.txt"
		f = open(filename, "w")
		for zi in range(self.n_t):
			word_prob = self.p_w_z[zi,:]
			word_index_prob = []
			for wi in range(self.n_w):
				word_index_prob.append([wi, word_prob[wi]])
			word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True)
			f.write("-------------\n" + "Topic #" + str(zi) + ":\n")
			for wi in range(topk):
				index = word_index_prob[wi][0]
				prob = word_index_prob[wi][1]
				f.write(self.corpus.vocabulary[index] + " " + str(prob) + "\n")
		f.close()

	def train(self):
		print "Training..."
		for i_iter in range(self.max_iter):

			# likelihood
			self.L = self.log_likelihood()
			
			self.print_top_words(10)
			
			print "Iter " + str(i_iter) + ", L=" + str(self.L)

			print "E-Step..."
			for di in range(self.n_d):
				for wi in range(self.n_w):
					sum_zk = np.zeros([self.n_t], dtype = float)
					for zi in range(self.n_t):
						sum_zk[zi] = self.p_z_d[di, zi] * self.p_w_z[zi, wi]
					sum1 = np.sum(sum_zk)
					if sum1 == 0:
						sum1 = 1
					for zi in range(self.n_t):
						self.p_z_dw[di, wi, zi] = sum_zk[zi] / sum1

			print "M-Step..."
			# update P(z|d)
			for di in range(self.n_d):
				for zi in range(self.n_t):
					sum1 = 0.0
					sum2 = 0.0
					for wi in range(self.n_w):
						sum1 = sum1 + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
						sum2 = sum2 + self.n_w_d[di, wi]
					if sum2 == 0:
						sum2 = 1
					self.p_z_d[di, zi] = sum1 / sum2

			# update P(w|z)
			for zi in range(self.n_t):
				sum2 = np.zeros([self.n_w], dtype = np.float)
				for wi in range(self.n_w):
					for di in range(self.n_d):
						sum2[wi] = sum2[wi] + self.n_w_d[di, wi] * self.p_z_dw[di, wi, zi]
				sum1 = np.sum(sum2)
				if sum1 == 0:
					sum1 = 1
				for wi in range(self.n_w):
					self.p_w_z[zi, wi] = sum2[wi] / sum1

def main(argv):
    print "Usage: python ./plsa.py <number_of_topics> <maxiteration>"

    # load stop words
    load_stop_words("./data/stopwords.txt")

    corpus = Corpus()
    document_paths = ['./data/events_2010/GulfOilSpill/', './data/events_2010/HaitiEarthquake/',
                      './data/events_2010/MichaelJacksonDied/', './data/events_2010/PakistanFloods/',
                      './data/events_2010/RussianForestFires/', './data/events_2010/SouthAfricaWorldCup/']

    #document_paths = ['./texts/grimm_fairy_tales', './texts/tech_blog_posts', './texts/nyt']
    
    for document_path in document_paths:
        for document_file in glob.glob(os.path.join(document_path, '*.txt')):
            document = Document(document_file)
            document.split(STOP_WORDS_DIC)
            corpus.add_document(document)
	corpus.build_vocabulary()
    

    number_of_topics = 12 #int(argv[1])
    max_iterations = 100 #int(argv[2])
    plsa = Plsa(corpus, number_of_topics, max_iterations, "./model/")
    plsa.train()   

if __name__ == "__main__":
    main(sys.argv)



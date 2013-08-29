# -*- coding: utf-8 -*-
'''
lipiji.sdu@gmail.com

original code from: https://github.com/hitalex/PLSA
'''

import sys
import os
import glob
import re
import numpy as np
from random import random
from operator import itemgetter # for sort

def normalize(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0
    """
    if abs(s) < 1e-6:
        print "Sum of vectors sums almost to 0. Stop here."
        print "Vec: " + str(vec) + " Sum: " + str(s)
        assert(0) # assertion fails
    """
        
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
        
    def plsa(self, number_of_topics, max_iter):

        '''
        Model topics.
        '''
        # Get vocabulary and number of documents.
        self.build_vocabulary()
        number_of_documents = len(self.documents)
        vocabulary_size = len(self.vocabulary)

        print "Vocabulary size:" + str(vocabulary_size)
        print "Number of documents:" + str(number_of_documents)

        print "EM iteration begins..."
        # build term-doc matrix
        term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
        for d_index, doc in enumerate(self.documents):
            term_count = np.zeros(vocabulary_size, dtype = np.int)
            for word in doc.words:
                if word in self.vocabulary:
                    w_index = self.vocabulary.index(word)
                    term_count[w_index] = term_count[w_index] + 1
            term_doc_matrix[d_index] = term_count

        # Create the counter arrays.
        self.document_topic_prob = np.zeros([number_of_documents, number_of_topics], dtype=np.float) # P(z | d)
        self.topic_word_prob = np.zeros([number_of_topics, len(self.vocabulary)], dtype=np.float) # P(w | z)
        self.topic_prob = np.zeros([number_of_documents, len(self.vocabulary), number_of_topics], dtype=np.float) # P(z | d, w)

        # Initialize
        print "Initializing..."
        # randomly assign values
        self.document_topic_prob = np.random.random(size = (number_of_documents, number_of_topics))
        for d_index in range(len(self.documents)):
            normalize(self.document_topic_prob[d_index]) # normalize for each document
        self.topic_word_prob = np.random.random(size = (number_of_topics, len(self.vocabulary)))
        for z in range(number_of_topics):
            normalize(self.topic_word_prob[z]) # normalize for each topic

        # Run the EM algorithm
        for iteration in range(max_iter):
            print "Iteration #" + str(iteration + 1) + "..."
            print "E step:"
            for d_index, document in enumerate(self.documents):
                for w_index in range(vocabulary_size):
                    prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
                    if sum(prob) == 0.0:
                        print "d_index = " + str(d_index) + ",  w_index = " + str(w_index)
                        print "self.document_topic_prob[d_index, :] = " + str(self.document_topic_prob[d_index, :])
                        print "self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index])
                        print "topic_prob[d_index][w_index] = " + str(prob)
                        exit(0)
                    else:
                        normalize(prob)
                    self.topic_prob[d_index][w_index] = prob
            print "M step:"
            # update P(w | z)
            for z in range(number_of_topics):
                for w_index in range(vocabulary_size):
                    s = 0
                    for d_index in range(len(self.documents)):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.topic_word_prob[z][w_index] = s
                normalize(self.topic_word_prob[z])
            
            # update P(z | d)
            for d_index in range(len(self.documents)):
                for z in range(number_of_topics):
                    s = 0
                    for w_index in range(vocabulary_size):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.document_topic_prob[d_index][z] = s
#                print self.document_topic_prob[d_index]
#                assert(sum(self.document_topic_prob[d_index]) != 0)
                normalize(self.document_topic_prob[d_index])

def print_topic_word_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print topic-word distribution to file and list @topk most probable words for each topic
    """
    print "Writing topic-word distribution to file: " + filepath
    V = len(corpus.vocabulary) # size of vocabulary
    assert(topk < V)
    f = open(filepath, "w")
    for k in range(number_of_topics):
        word_prob = corpus.topic_word_prob[k, :]
        word_index_prob = []
        for i in range(V):
            word_index_prob.append([i, word_prob[i]])
        word_index_prob = sorted(word_index_prob, key=itemgetter(1), reverse=True) # sort by word count
        f.write("Topic #" + str(k) + ":\n")
        for i in range(topk):
            index = word_index_prob[i][0]
            f.write(corpus.vocabulary[index] + " ")
        f.write("\n")
        
    f.close()
    
def print_document_topic_distribution(corpus, number_of_topics, topk, filepath):
    """
    Print document-topic distribution to file and list @topk most probable topics for each document
    """
    print "Writing document-topic distribution to file: " + filepath
    assert(topk < number_of_topics)
    f = open(filepath, "w")
    D = len(corpus.documents) # number of documents
    for d in range(D):
        topic_prob = corpus.document_topic_prob[d, :]
        topic_index_prob = []
        for i in range(number_of_topics):
            topic_index_prob.append([i, topic_prob[i]])
        topic_index_prob = sorted(topic_index_prob, key=itemgetter(1), reverse=True)
        f.write("Document #" + str(d) + ":\n")
        for i in range(topk):
            index = topic_index_prob[i][0]
            f.write("topic" + str(index) + " ")
        f.write("\n")
        
    f.close()        

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
    
    

    number_of_topics = 10 #int(argv[1])
    max_iterations = 50 #int(argv[2])
    corpus.plsa(number_of_topics, max_iterations)
        

    print_topic_word_distribution(corpus, number_of_topics, 10, "./model/topic-word.txt")
    print_document_topic_distribution(corpus, number_of_topics, 10, "./model/document-topic.txt")

if __name__ == "__main__":
    main(sys.argv)



#Vanilla Recurrent Neural Network
#Our goal is to build a Language Model using a Recurrent Neural Network

#Data: Comments from Raddit.

import tensorflow as tf
import numpy as np
import nltk
import csv
import itertools

source_data_file = "../Data/reddit-comments.csv"

#---------------------------------------
#Tokenize Raw Text using nltk
print "Reading CSV files..."

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

with open(source_data_file,'rb') as f:
    reader = csv.reader(f, skipinitialspace = True)
    #Skip first line/header
    reader.next()
    token_sentences = []
    for row in reader:
        #row is a list of string, since we didn't specify delimiter so whole line stored in row[0]
        token_sentences += nltk.sent_tokenize(row[0].decode('utf-8').lower())
    sentences = ["%s %s %s" % (sentence_start_token, sentence, sentence_end_token) for sentence in token_sentences]
print "Parsed %d sentences." % (len(sentences))


#Tokenize the sentences into words
#sentence is a sentence string, word_tokenize(sentence) return a list of words. tokenized_sentence is nested list
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Count the word frequencies
#FreqDist return the dictionary of frequency distribution
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

#Get most common words and build index_to_word and word_to_vector vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([word,index] for index,word in enumerate(index_to_word))
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sentence in enumerate(tokenized_sentences):
    #tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]
    for j,word in enumerate(sentence):
        if word in word_to_index:
            continue
        else:
            sentence[j] = unknown_token
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
#---------------------------------------


#---------------------------------------
#Prepare X_train and Y_train
X_train_list = []
for sentence in token_sentences:
    x = [word_to_index[word] for word in sentence[:-1]]
    X_train_list.append()
X_train = np.asarray(X_train_list)

Y_train_list = []
for sentence in token_sentences:
    y = [word_to_index[word] for word in sentence[1:]]
    Y_train_list.append()
Y_train = np.asarray(Y_train_list)
print "\n Prepare Training Data Finished!"

#---------------------------------------
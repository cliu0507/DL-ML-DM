#Vanilla Recurrent Neural Network
#Our goal is to build a Language Model using a Recurrent Neural Network

#Data: Comments from Raddit.

import tensorflow as tf
import numpy as np
import csv
import itertools
import nltk

#source_data_file = "../Data/reddit-comments.csv"
source_data_file = "../Data/short.csv"

#---------------------------------------
#Tokenize Raw Text using nltk
print "Reading CSV files..."

#vocabulary_size = 8000
vocabulary_size = 100
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
        if word not in word_to_index:
            tokenized_sentences[i][j] = unknown_token
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
#---------------------------------------


#---------------------------------------
#Prepare X_train and Y_train
X_train_list = []
for sentence in tokenized_sentences:
    x = [word_to_index[word] for word in sentence[:-1]]
    X_train_list.append(x)
X_train = np.asarray(X_train_list)

Y_train_list = []
for sentence in tokenized_sentences:
    y = [word_to_index[word] for word in sentence[1:]]
    Y_train_list.append(y)
Y_train = np.asarray(Y_train_list)
print "\n Prepare Training Data Finished!"
#---------------------------------------


#---------------------------------------
#Prepare one-hot nested vector:
X_train_one_hot_list = []
Y_train_one_hot_list = []

for sentence in X_train:
    sent_one_hot = np.zeros((len(sentence),vocabulary_size))
    for i, word_index in enumerate(sentence):
        sent_one_hot[i][word_index] = 1
    X_train_one_hot_list.append(sent_one_hot)
X_train_one_hot= np.array(X_train_one_hot_list)

for sentence in Y_train:
    sent_one_hot = np.zeros((len(sentence),vocabulary_size))
    for i, word_index in enumerate(sentence):
        sent_one_hot[i][word_index] = 1
    Y_train_one_hot_list.append(sent_one_hot)
Y_train_one_hot = np.array(Y_train_one_hot_list)
#---------------------------------------


#---------------------------------------
#Vanilla RNN Model Architecture
'''
S(t) = tanh(U*X(t) + W*S(t-1))
O(t) = softmax(V * S(t))
vocabulary size C = 8000 and a hidden layer size H = 100

Dimension/shape:
X(t) -> [8000,]
O(t) -> [8000,]
S(t) -> [100,]
U    -> [100,8000]
V    -> [100,8000] --- based on doc, it should be [8000,100]
W    -> [100,100]

'''
#---------------------------------------


#---------------------------------------
#TensorFlow Graph

num_hidden_state = 100
learning_rate = 0.001

#tf graph input
x = tf.placeholder("float", [None,vocabulary_size])
y = tf.placeholder("float", [None,vocabulary_size])
init_state = tf.zeros([num_hidden_state])

#Layer Weight/bias
U = tf.Variable(tf.random_normal([num_hidden_state,vocabulary_size]))
W = tf.Variable(tf.random_normal([num_hidden_state,num_hidden_state]))
#V = tf.Variable(tf.random_normal([vocabulary_size,num_hidden_state]))
V = tf.Variable(tf.random_normal([num_hidden_state,vocabulary_size]))

with tf.variable_scope('rnn_cell'):
    U = tf.get_variable('U', [num_hidden_state,vocabulary_size])
    W = tf.get_variable('W', [num_hidden_state,num_hidden_state])
    #V = tf.get_variable('V', [vocabulary_size,num_hidden_state])
    V = tf.get_variable('V', [num_hidden_state,vocabulary_size])

def rnn_cell(x_step, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        U = tf.get_variable('U', [num_hidden_state, vocabulary_size])
        W = tf.get_variable('W', [num_hidden_state, num_hidden_state])
        #V = tf.get_variable('V', [vocabulary_size, num_hidden_state])
        V = tf.get_variable('V', [num_hidden_state,vocabulary_size])
    return tf.tanh(tf.add(tf.matmul(U,x_step), tf.matmul(W,state)))

state = init_state
rnn_outputs = []
#x_list = tf.unpack(x, axis=0)
x.get_shape()
batch_size = tf.shape(x)[0]

for x_step in tf.shape(x)[0]:
    state = rnn_cell(x_step,state)
    rnn_outputs.append(state)
#IDEA: need to get all steps' output, contributing to total loss(Not only final output)
s=np.asarray(rnn_outputs)

pred  = tf.nn.softmax(tf.matmul(s,V))

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)


#---------------------------------------

#---------------------------------------
#Launch the graph
training_epochs = 5

init = tf.initialize.all_variables()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        #Training cycle
        avg_cost = 0
        for i in range(len(X_train_one_hot)):
            _, c = sess.run([optimizer, cost], feed_dict={x: X_train_one_hot[i], y:X_train_one_hot[i]})
            avg_cost_per_word = c/len(X_train_one_hot[i])
            #avg_cost += c/len(X_train_one_hot)
        print("Sentence:", '%05d' % (i), "cost=", "{:.09f}".format(avg_cost_per_word))

print("Opimization Finished!")
#---------------------------------------

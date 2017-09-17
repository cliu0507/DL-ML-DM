#RNN to predict Stock Market

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
'''
Assume we have 
4000 data points
400 features

Input

Feature Matrix:
np.array [4000,400]

Label Matrix:
np.array [4000,2]


Step = 5 
batch = 100
[800,5,400]
8 batches
x:8 * [100, 5, 400]
y:8 * [100, 2]

'''

# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
BATCH_SIZE = 100
num_features = 3
num_classes = 2
state_size = 4
learning_rate = 0.1
regularization = 0.0001


def gen_Data(BATCH_SIZE,num_features,num_classes,record_count=50000):
	feature_matrix = np.zeros((record_count,num_features),dtype=np.int8)
	label_matrix = np.zeros((record_count,num_classes),dtype=np.int8)
	for i, row in enumerate(feature_matrix):
		for j, feature in enumerate(row):
			feature_matrix[i,j] = randint(0,1)

	#Define the Pattern between feature and label
	'''
	example:
	only 
	if consecutive two days feature[0] = 1  => label of next day will be [0,1] 
	or 
	if consecutive two days feature[1] = 0  => label of next day will be [0,1]
	or 
	if consecutive two days feature[2] = 1 => label of next day will be [0,1]
 	'''
	for i, row in enumerate(label_matrix):
		if i < 2:
			label_matrix[i,0] = 1
		else:
			
			if feature_matrix[i-1,0] == 1 and feature_matrix[i-2,0] == 1 \ 
			or feature_matrix[i-1,1] == 0 and feature_matrix[i-2,1] == 0 \
			or (i>=3 and feature_matrix[i-3,2] == 1 and feature_matrix[i-2,2] == 1 and feature_matrix[i-1,2] == 1):
				#Satisify pattern
				label_matrix[i,1] = 1
			
			'''
			if feature_matrix[i-1,0] == 1 and feature_matrix[i-2,0] == 1:
				label_matrix[i,1] = 1
			'''
			else:
				#Otherwise assign it as 0
				label_matrix[i,0] = 1
	return feature_matrix,label_matrix



def gen_Pass(feature_matrix,label_matrix, BATCH_SIZE, num_steps,num_features,num_classes,offset = 0):
	
	features_matrix_copy = (np.array(feature_matrix,copy = True)[offset:-1,:]).flatten()
	label_matrix_copy = (np.array(label_matrix,copy = True)[offset+1:,:]).flatten()
	#print len(features_matrix_copy)
	num_batch = len(features_matrix_copy)/(num_features * num_steps * BATCH_SIZE)
	if num_batch <= 0:
		raise ValueError('num_batch is smaller than or equal 0!')
	#feature : [num_batch, BATCH_SIZE, num_steps, num_features]
	feature = features_matrix_copy[:num_batch * num_features * num_steps * BATCH_SIZE].reshape((-1,BATCH_SIZE,num_steps,num_features))

	#label : [num_batch, BATCH_SIZE , num_classes]
	label = label_matrix_copy[:num_batch * num_classes * num_steps * BATCH_SIZE].reshape((-1,BATCH_SIZE,num_steps,num_classes))[:,:,-1,:]
	return feature,label


#---------------------------------------------
'''
Define Neural Network:
'''
#Placeholder for input x and y
x = tf.placeholder(tf.float32, [None, num_steps, num_features], name='input_placeholder')
y = tf.placeholder(tf.float32, [None, num_classes], name='labels_placeholder')
#init_state = tf.zeros([BATCH_SIZE, state_size])
init_state = tf.placeholder(tf.float32, [None, state_size])

"""
RNN
"""
rnn_inputs = tf.unpack(x, axis=1)

cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

"""
Predictions, loss, training step
"""

with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [state_size, num_classes])
	b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

#logit : [100,2]
logit = tf.matmul(rnn_outputs[-1], W) + b 

#prediction: [100,2] - for last step
prediction = tf.nn.softmax(logit)

#for all 5 steps (These are list of logit and prediction of all steps)
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]


#losses = [100]
losses = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logit)
total_loss = tf.reduce_mean(losses + regularization * tf.nn.l2_loss(W))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

#---------------------------------------------


#Generate Training Data
#feature_matrix_test: [num_data_points,num_feature]
#label_matrix_test : [num_data_points, num_classes]
feature_matrix,label_matrix= gen_Data(BATCH_SIZE,num_features,num_classes)


"""
Function to train the network
"""
print "Training Starts..."
print feature_matrix.shape
print label_matrix.shape
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	training_losses = []
	for epoch in range(0,10):
		for offset in range(0,num_steps):
			#X_list : [num_batch, BATCH_SIZE, num_steps, num_feature]
			#Y_list : [num_batch, BATCH_SIZE, num_classes] (Note, only keep the y value of last step)
			X_list,Y_list= gen_Pass(feature_matrix, label_matrix, BATCH_SIZE, num_steps,num_features,num_classes,offset) 
			for batch_id,(X, Y) in enumerate(zip(X_list,Y_list)):
				training_loss = 0
				training_state = np.zeros((BATCH_SIZE, state_size))
				#print X.shape
				#print Y.shape
				tr_losses, training_loss_, training_state, _ = \
					sess.run([losses,total_loss,final_state,train_step],
						feed_dict={x:X, y:Y, init_state:training_state})
				training_loss += training_loss_
				training_losses.append(float(training_loss)/BATCH_SIZE)
				if batch_id % 30 == 0:
					print "Batch Training Loss: " + str(float(training_loss)/BATCH_SIZE)
			print "Finish One Pass of Data"
		print "Epoch Finished...Start Another Epoch"
	print "Training Finished"
	#plt.plot(training_losses)
	#plt.show()
	

	'''
	This is the default validation method, basically check every num_steps steps
	#Calculate Accuracy
	print "Validation in interval as num_steps..."
	#Generate Test Data Set
	#feature_matrix_test: [num_data_points,num_feature]
	#label_matrix_test : [num_data_points, num_classes]
	feature_matrix_test,label_matrix_test= gen_Data(BATCH_SIZE,num_features,num_classes,10000)
	
	#X_test_list : [num_batch, BATCH_SIZE, num_steps, num_feature]
	#Y_test_list : [num_batch, BATCH_SIZE, num_classes] (Note, only keep the y value of last step)
	#Also do the callibrate for y and x (Note: x(t-4), x(t-3)...x(t) leads to a value in y(t+1))
	X_test_list,Y_test_list= gen_Pass(feature_matrix_test, label_matrix_test, BATCH_SIZE, num_steps,num_features,num_classes,offset) 
	print np.array(X_test_list).shape
	TOTAL_CASE = 0
	TOTAL_CORRECT_CASE = 0
	for (X, Y) in zip(X_test_list,Y_test_list):
		training_loss = 0
		training_state = np.zeros((BATCH_SIZE, state_size))

		#Run the inference
		predictions_, prediction_, training_state = \
			sess.run([predictions,prediction,final_state],
				feed_dict={x:X, init_state:training_state})
		
		#print prediction_.argmax(axis=1)[:50]
		#print Y.argmax(axis=1)[:50]
		#Compare ground truth and prediction
		TOTAL_CORRECT_CASE += int(np.sum(prediction_.argmax(axis=1) == Y.argmax(axis=1)))
		TOTAL_CASE += len(Y)
		#print np.array(X).flatten()[:100]
		#print np.array(prediction_).flatten().reshape((-1,num_classes)).argmax(axis=1)[:100]
	if TOTAL_CASE == 0:
		raise ValueError("TEST SET is empty!")
	print "Another Validataion method: The overall accuracy: %d / %d" %(TOTAL_CORRECT_CASE,TOTAL_CASE) + " ==> " + str(float(TOTAL_CORRECT_CASE)/TOTAL_CASE)	
	'''

	#Start to do prediction
	print "\n Validation in place..."
	feature_matrix_ver,label_matrix_ver= gen_Data(BATCH_SIZE,num_features,num_classes,10000)
	x_ver = feature_matrix_ver.flatten() #length = 1000 * num_features
	y_ver = label_matrix_ver.flatten() #length = 1000 * num_classes

	#Reshape x, y, feed into placeholder of neural network
	#Pick the first batch
	TEST_BATCH_SIZE = 10*BATCH_SIZE
	x_ = x_ver.reshape((-1,num_steps,num_features))[:TEST_BATCH_SIZE]
	y_ = y_ver.reshape((-1,num_steps,num_classes))[:TEST_BATCH_SIZE]

	training_state = np.zeros((TEST_BATCH_SIZE, state_size))
	predictions_ = \
			sess.run(predictions,
				feed_dict={x:x_, init_state:training_state})
	
	y_prediction = []
	for row in range(0,TEST_BATCH_SIZE):
		for step in range(0,num_steps):
			y_prediction.append(predictions_[step][row])

	print "Feature_matrix:"
	print feature_matrix_ver[:20]
	print "Groundtruth:"
	print y_.reshape((-1,num_classes)).argmax(axis=1)[1:21]
	print "Prediction:"
	print np.array(y_prediction).argmax(axis=1)[:20]
	TOTAL_CORRECT_CASE = int(np.sum(np.array(y_prediction).argmax(axis=1)[:-1] == y_.reshape((-1,num_classes)).argmax(axis=1)[1:]))
	TOTAL_CASE = len(y_prediction)-1 
	print "The overall accuracy: %d / %d" %(TOTAL_CORRECT_CASE,TOTAL_CASE) + " ==> " + str(float(TOTAL_CORRECT_CASE)/TOTAL_CASE)	

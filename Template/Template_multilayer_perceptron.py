#Use Tensorflow to solve machine learning problem
#Multilayer Perceptron

#Import tensorflow library
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#---------------------------------------------
#Input Data Module:
#x_training is feature matrix of training set
#y_training is label matrix of training set
#x_test is feature matrix of training set
#y_test is label matrix of training set

x_training = []
y_training = []

x_test = []
y_test = []

##############################################
#Generate a random n-class classification problem by Sklearn
X, Y = make_classification(n_samples=1000, n_features=10, n_classes=3,n_clusters_per_class =1,n_informative = 5)
#X, Y = make_classification(n_samples=1000, n_features=10, n_classes=2,n_clusters_per_class =1)


x_training = X[:500][:]
x_test = X[500:][:]

Y_Sparse = np.zeros((X.shape[0],5))
for i in range(Y_Sparse.shape[0]):
    Y_Sparse[i][Y[i]] = 1

print Y_Sparse
print Y

y_training = Y_Sparse[:500][:]
y_test = Y_Sparse[500:][:]
##############################################

num_feature = x_training.shape[1]
num_class = y_training.shape[1]
num_training_sample = x_training.shape[0]
num_test_sample = x_test.shape[0]

print("Training Set Size %d" % num_training_sample)
print("Test Set Size %d" % num_test_sample)

'''
Example Case:
feature number: 10
multiple classes, class number : 3

training set sample number: 100
test set sample number : 100

Data Sample: feature vector: [2,5,6,7,7,6,7,8,3,-1], label vector: [1,0,0]
'''
#---------------------------------------------


#---------------------------------------------
#Build Model Module:

# Parameters
learning_rate = 0.03
training_epochs = 40000
display_step = 1000

# Network Parameters
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = num_feature # MNIST data input (img shape: 28*28)
n_classes = num_class # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#---------------------------------------------


#---------------------------------------------
#Model Training Module:

#Create session
sess = tf.InteractiveSession()

# Initializing the variables Weights list
init = tf.initialize_all_variables()

#Initial run
sess.run(init)

# Launch the graph
# Training cycle
for epoch in range(training_epochs):
    # Run optimization op (backprop) and cost op (to get loss value)
    _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_training, y: y_training})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_val))
print("MLP Optimization Finished!")
final_cost = sess.run(cost, feed_dict= {x:x_training , y:y_training})
print("Final Training Cost:" "{:.9f}".format(final_cost) , " Weights:", sess.run(weights), " Bias:", sess.run(biases))
#---------------------------------------------


#---------------------------------------------
#Test/Validation Module (Version One):
print("Testing Test Data Set:")
weights = sess.run(weights)
bias = sess.run(biases)
pred_val = sess.run(pred,feed_dict={x:x_test})
#correction_prediction numpy list example: [True, False, False, True] --- size is the number of test samples
correction_prediction = np.equal(np.argmax(pred_val,1),np.argmax(y_test,1))
accuracy = 0
for i in range(correction_prediction.size):
  if correction_prediction[i]:
    accuracy += float(1)/float(correction_prediction.size)
print("Test Set Prediction Accuracy(Version 1): %f" % accuracy)
#---------------------------------------------


#---------------------------------------------
#Test/Validation Module (Version Two):
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_val = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
print("Test Set Prediction Accuracy(Version 2): %f" % accuracy_val)
#---------------------------------------------
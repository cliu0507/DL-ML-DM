#Use Tensorflow to solve machine learning problem
#Logistic Regression (Binary Classification)

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
X, Y = make_classification(n_samples=1000, n_features=10, n_classes=2,n_clusters_per_class =1)

x_training = X[:500][:]
x_test = X[500:][:]


# X is [1000, n_feature] matrix, Y is [1000,] vector, in logistic regression, we don't need to format Y, which is different from softmax regression
#print Y_Sparse
print Y

y_training = Y[:500]
y_test = Y[500:]
##############################################

num_feature = x_training.shape[1]
num_class = 2
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
learning_rate = 0.5
training_epochs = 30000
display_epochs = 1000

#tensor for feature vector x , shape is [None,feature_num] -- matrix
x = tf.placeholder(tf.float32, [None,num_feature])

#tensor for logistic regression weights, shape is [feature_num] --vector
W = tf.Variable(tf.zeros([num_feature],dtype=tf.float32))

#tensor for bias weight, shape is [1] -- float
b = tf.Variable(tf.zeros([],tf.float32))

#prediction tensor y, shape is [None,1] --- broadcasting feature of numpy
xW = tf.add(tf.reduce_sum(tf.mul(x,W),1), b)
#xW_test = tf.mul(x,W)

#tensor of ground truth label vectors
y = tf.placeholder(tf.float32,[None])

#hypothesis tensor -- h is [None,1]
h = 1/(tf.exp(-xW) + 1)

#cost function (cross entropy)
cost = -tf.reduce_mean(y * tf.log(h) + (1-y) * tf.log(1 - h))

#Set GradientDescentOptimizer with learning rate 0.5, to minimize cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#---------------------------------------------


#---------------------------------------------
#Model Training Module:

#Create session
sess = tf.InteractiveSession()

#Initialize all variables: W matrix and Bias
init = tf.initialize_all_variables()

sess.run(init)

for epoch in range(training_epochs):

    #Training Process
    _,cost_val,weights,bias = sess.run([optimizer,cost,W, b], feed_dict={x:x_training, y:y_training})

    #Print epoch training status on console
    if epoch % display_epochs == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", cost_val)

print('Logistic Regression Optimization Finished!')
final_cost = sess.run(cost, feed_dict = {x:x_training,y:y_training})
print("Final Training Cost:" "{:.9f}".format(final_cost) , " Weights:", sess.run(W), " Bias:", sess.run(b))
#---------------------------------------------


#---------------------------------------------
#Test/Validation Module (Version One):
print("Testing Test Data Set:")
weights = sess.run(W)
bias = sess.run(b)
h = sess.run(h,feed_dict={x:x_test})

for index in range(h.size):
    if h[index] >= 0.5:
        h[index] = 1
    else:
        h[index] = 0

correction_prediction = np.equal(h,y_test)
accuracy = 0
for i in range(correction_prediction.size):
  if correction_prediction[i]:
    accuracy += float(1)/float(correction_prediction.size)
print("Test Set Prediction Accuracy(Version 1): %f" % accuracy)

#correction_prediction numpy list example: [True, False, False, True] --- size is the number of test samples
#---------------------------------------------
#Use Tensorflow to solve machine learning problem
#Softmax Regression

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
learning_rate = 0.5
training_epochs = 100000
display_epochs = 1000

#tensor for feature vector x, shape is [None,feature_num]
x = tf.placeholder(tf.float32, [None, num_feature])

#tensor for softmax regression weights, shape is [feature_num, class_num]
W = tf.Variable(tf.zeros([num_feature,num_class]))

#tensor for bias weight , shape is [class_num]
b = tf.Variable(tf.zeros([num_class]))

#prediction tensor y, shape is [None,3], note xW should be x*W + b
xW = tf.matmul(x,W) + b

#tensor of label vectors
y = tf.placeholder(tf.float32, [None, num_class])

# The raw formulation of cross-entropy is:
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.softmax(xW)),reduction_indices=[1]))
# But it can be numerically unstable.
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'xW', and then average across the batch.

'''
Below is the details of softmax regression implementation
Softmax regression: http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

We could use Raw formulation instead of calling tf.nn.softmax_cross_entropy_with_logits(xW, y):
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.softmax(xW)),reduction_indices=[1]))

tf.nn.softmax(logits, dim=-1, name=None) Computes log softmax activations.

For each batch i and class j we have
softmax = exp(logits) / reduce_sum(exp(logits), dim)

logits: A non-empty Tensor
dim: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
Suppose that num_class = 3, num_feature = 100
xW is tensor with shape [None, 3]
y is tensor with shape [None, 3]
exp(xW) is tensor with shape [None, 3]
reduce_sum(exp(xW),1) is tensor with shape [None, 1]  (Add up values along row-wise)
exp(xW)/reduce_sum(exp(xW),1) is tensor with shape [None, 3] (numpy boardcasting feature)
tf.log(exp(xW)/reduce_sum(exp(xW),1)) is tensor with shape [None, 3]

Example: xW_example has shape [ 2, 3] --- two samples, three classes
         xW_example = [[-2,4,6],
                            [4,-5,2]
                            ]
         y = [[1,0,0],
              [0,1,0]
              ]

         tf.log(tf.softmax(xW)) = tf.log([
                                    [exp(-2)/(exp(-2)+exp(4)+exp(6)) , exp(4)/(exp(-2)+exp(4)+exp(6)), exp(6)/(exp(-2)+exp(4)+exp(6))],
                                    [exp(4)/exp(4)+exp(-5)+exp(2) , exp(-5)/exp(4)+exp(-5)+exp(2) ,exp(2)/exp(4)+exp(-5)+exp(2) ]
                                    ])
                                    = tf.log([[0.01, 0.05 , 0.94],[0.9,0.01,0.09] ])
                                    = [[-6.64,-4.3,-0.08 ], [-0.15, -6.6,-3.5]]
         y * tf.log(tf.softmax(xW)) = [[1,0,0],[0,1,0]] * [[-6.64,-4.3,-0.08 ], [-0.15, -6.6,-3.5]]
                                        = [[-0.64,0,0] , [0,-6.6,0]] -----this is not matrxi multiplication
                                        #Note: matrix a.shape = [2,3] , b.shape = [2,3] ===> (a*b).shape = [2,3] ===>numpy array broadcasting
         tf.reduce_sum(y * tf.log(tf.softmax(xW)),reduction_indices=[1]) = tf.reduce_sum([[-0.64,0,0] , [0,-6.6,0]],reduction_indices = [1])
                                        = [[-0.64], [-0.66]] ----> shape is [2,1]
         tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.softmax(xW)),reduction_indices=[1]))
                                    = tf.reduce_mean([[0.64],[0.66]])
                                    = 0.65
'''


'''
suppose number of classes is 3
xW = [None,3]
y = [None,3]
tf.nn.softmax_cross_entropy_with_logits(xW, y)'s shape is [None,3],

example:
tf.nn.softmax_cross_entropy_with_logits(xW, y) looks like example: (-1) *[[-0.64,0,0] , [0,-6.6,0]]
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(xW, y)) = 0.65
'''

#Cost Function (cross entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(xW, y))

#Set GradientDescentOptimizer with learning rate 0.5, to minimize cross_entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#---------------------------------------------


#---------------------------------------------
#Model Training Module:

#Create session
sess = tf.InteractiveSession()

#Initialize all variables: W matrix (Note: all variables must be initialized before calling session.run())
init = tf.initialize_all_variables()

sess.run(init)

for epoch in range(training_epochs):
  #Training:
  _,cost,weights,bias = sess.run([optimizer,cross_entropy,W,b], feed_dict={x:x_training , y:y_training})

  #Print epoch training status on console
  if epoch % display_epochs == 0:
    # An alternative Way to print:
    # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost), "W=", sess.run(W), "b=", sess.run(b))
    #print("Epoch:", '%04d' % (epoch+1), "cost=", cost, "W=", weights, "b=",bias)
    print("Epoch:", '%04d' % (epoch + 1), "cost=", cost)

print("Softmax Regression Optimization Finished!")
final_cost = sess.run(cross_entropy, feed_dict= {x:x_training , y:y_training})
print("Final Training Cost:" "{:.9f}".format(final_cost) , " Weights:", sess.run(W), " Bias:", sess.run(b))
#---------------------------------------------


#---------------------------------------------
#Test/Validation Module (Version One):
print("Testing Test Data Set:")
weights = sess.run(W)
bias = sess.run(b)
xW_val = sess.run(xW,feed_dict={x:x_test})
#correction_prediction numpy list example: [True, False, False, True] --- size is the number of test samples
correction_prediction = np.equal(np.argmax(xW_val,1),np.argmax(y_test,1))
accuracy = 0
for i in range(correction_prediction.size):
  if correction_prediction[i]:
    accuracy += float(1)/float(correction_prediction.size)
print("Test Set Prediction Accuracy(Version 1): %f" % accuracy)

#---------------------------------------------


#---------------------------------------------
#Test/Validation Module (Version Two):
correct_prediction = tf.equal(tf.argmax(xW, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_val = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
print("Test Set Prediction Accuracy(Version 2): %f" % accuracy_val)

#---------------------------------------------

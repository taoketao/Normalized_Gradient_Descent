'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from param_searcher import get_params

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#if not len(sys.argv)==3:
#    raise Exception
#seed=int(sys.argv[1])
#params_filename = sys.argv[2]
#
#print('training network with seed <'+str(seed)+'>.')
#tf.set_random_seed(seed)
args = get_params()

# Set Parameters
learning_rate = 0.001
learning_rate = args['lr']
training_epochs = 1
batch_size = 16
display_step = 1

# Network Parameters
num_layers = len(args['net type'])
n_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = args["layer sizes"][0]
layer_sizes = [n_input] 
for i in range(num_layers-1):
    n_hidden_i = args["layer sizes"][i]
    layer_sizes.append(n_hidden_i)
n_classes = 10 # MNIST total classes (0-9 digits)
layer_sizes.append(n_classes)
print(layer_sizes)

opt = args['opt']
if not args['base normalization']=='meansum':
    raise Exception("NotImplemented")

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
#nrm = tf.placeholder("float32", name='norm')

# Create model
def multilayer_perceptron(x, weights, biases):
    layer = x
    for lyr_id in range(num_layers):
        layer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer = tf.contrib.layers.batch_norm(layer)
        if lyr_id==num_layers-1:
            layer = tf.nn.sigmoid(layer)
        else:
            layer = tf.nn.relu(layer)
    return layer

# Store layers weight & bias
layer_vars = []
layer_shapes = []
weights = {}
biases = {}
for lyr_id in range(num_layers):
    shape = [layer_sizes[lyr_id], layer_sizes[lyr_id+1]]
    layer_shapes.append(shape)
    lastlayer = True if lyr_id==num_layers-1 else False
    i = 1.0 if lastlayer else 2.0 # relu: kill half
    if args['init var'][lyr_id]=='xav in':
        var_fctr = i/shape[0] 
    if args['init var'][lyr_id]=='glorot':
        var_fctr = 2*i/np.prod(shape)
    if args['init var'][lyr_id]=='sqrt in':
        var_fctr = 1/np.sqrt(i*np.prod(shape))
    layer_vars.append(var_fctr)

    if init_type=='unif':
        pass # calculate, given variance, what the bound should be

    ; # multiply variance bound by normalization scaling factor
    ; # shift variance bound to min=0 if RELU
    
    if lastlayer:
        weights['out'] = variable
    else:
        weights['h'+str(lyr_id+1)] = variable

print(layer_shapes, layer_vars)
sys.exit()


h1_sh = [n_input, n_hidden_1]
h1_var = 2*float(np.prod(h1_sh))**-0.5
fctrs.append(h1_var)
if num_layers>1:
    h2_sh = [n_hidden_1, n_hidden_2]
    h2_var = 2*float(np.prod(h2_sh))**-0.5
    fctrs.append(h2_var)
if num_layers>2:
    h3_sh = [n_hidden_2, n_hidden_3]
    h3_var = 2*float(np.prod(h2_sh))**-0.5
    fctrs.append(h3_var)

#outsh = [n_hidden_2, n_classes];    outftr = 2*float(np.prod(outsh))**-0.5
weights = {
    #'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'h1': tf.Variable(tf.random_normal(h1_sh, h1_ftr)),
    'h1': tf.Variable(tf.random_uniform(h1_sh, minval=-h1_ftr, maxval=h1_ftr)),
    'h2': tf.Variable(tf.random_uniform(h2_sh, minval=-h2_ftr, maxval=h2_ftr)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

if opt=='sgd':
    optimizer  = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
elif opt=='adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

grads = optimizer.compute_gradients(cost, gate_gradients=2)
#nrm = tf.abs(tf.reduce_sum([tf.reduce_sum(gr[1]) for gr in grads]))
nrm = tf.pow(tf.abs(tf.reduce_sum([tf.reduce_sum(gr[1]) for gr in grads])), 0.5)
nrm_grads = [(tf.divide(grad, nrm),var) for grad, var in grads]
updates = optimizer.apply_gradients(nrm_grads)


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
#C, S = [], []
t = time.gmtime()
savestr = './adam-ngd-out--'+str(t.tm_mon)+'-'+str(t.tm_mday)+'-'+\
        str(t.tm_year)+'--'+str(t.tm_hour)+'_'+str(t.tm_min)+'-seed_'+str(seed)+'.txt'
with open(savestr, 'w') as outfile:
  with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
#            g, c = sess.run([grads, cost], feed_dict={x: batch_x, y: batch_y})
            u, g, c = sess.run([updates, grads, cost], feed_dict={x: batch_x, y: batch_y})
            _sum = np.sum([np.sum(gr[1]) for gr in g])
            if i%100==0:
                # Test model
                if epoch==0:
                    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                    # Calculate accuracy
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    A = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
                    print("Accuracy:", A,end=', ')
                print(_sum)


#                print([gr[1].shape for gr in g])
#            u = sess.run([updates], feed_dict={nrm: _sum})
#            S.append(_sum)
#            C.append(c)
            outfile.write('S '+str(_sum)+'\n')
            outfile.write('C '+str(c)+'\n')
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0 or epoch < 3:

            # Test model
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            A = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost), "Accuracy:", A)

    print("Optimization Finished!")

print("Successfully saved run to <"+savestr+">.")
sys.exit()
plt.title('adam normalized gradient | cost red | sum of gradients: green | final accuracy: '+str(A))
plt.plot(S, c='green')
plt.plot(C, c='red')
plt.show()

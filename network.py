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
if not args['base normalization'] in ['meansum','thresh']:
    raise Exception("NotImplemented")

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
#nrm = tf.placeholder("float32", name='norm')

# Create model
def multilayer_perceptron(x, weights, biases):
    layer = x
    for lyr_id in range(len(weights)):
        layer = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer = tf.contrib.layers.batch_norm(layer)
        if lyr_id==num_layers-1:
            layer = tf.nn.sigmoid(layer)
        else:
            layer = tf.nn.relu(layer)
    return layer

layer_vars = []
layer_shapes = []
weights = {}
biases = {}
# Initialize random weights
for lyr_id in range(num_layers):
    lastlayer = True if lyr_id==num_layers-1 else False
    if lastlayer:
        shape = [layer_sizes[lyr_id], n_classes]
    else:
        shape = [layer_sizes[lyr_id], layer_sizes[lyr_id+1]]
    layer_shapes.append(shape)
    i = 1.0 if lastlayer else 2.0 # relu: kill half
    if args['init var'][lyr_id]=='xav in':
        var_fctr = i/shape[0] 
    if args['init var'][lyr_id]=='glorot':
        var_fctr = 2*i/np.prod(shape)
    if args['init var'][lyr_id]=='sqrt in':
        var_fctr = 1/np.sqrt(i*np.prod(shape))
    layer_vars.append(var_fctr)

    var_fctr *= args['weights scales'][lyr_id]

    if args['init type'][lyr_id]=='unif':
        r = np.sqrt(12 * var_fctr)
        if lastlayer: 
            variable = tf.Variable(tf.random_uniform(shape, -r*0.25, 1.75*r))
        else:
            variable = tf.Variable(tf.random_uniform(shape, -r, r))
    if args['init type'][lyr_id]=='trunc normal':
        if lastlayer:
            variable = tf.Variable(tf.random_normal(shape, 0, var_fctr))
        else:
            variable = tf.Variable(tf.random_normal(shape, var_fctr, var_fctr))

    weights['out' if lastlayer else 'h'+str(lyr_id+1)] = variable
    k = 'out' if lastlayer else 'b'+str(lyr_id+1)
    biases [k] = \
            tf.Variable(tf.random_normal(shape[1:2], var_fctr))
    print(variable.shape, biases[k].shape)
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
#if args['base normalization']=='thresh':
#    abssum = tf.abs(tf.reduce_sum([tf.reduce_sum(gr[1]) for gr in grads]))
#    # Multiply by normalizer nrm.
#    nrm = { 'sum num weights': (abssum)**-1 / tf.maximum(abssum, \
#                    np.sum(np.sum(s) for s in layer_shapes)),
#            'sqrt prod num weights': tf.maximum(np.sqrt(np.prod(s[0]*s[1] \
#                    for s in layer_shapes), abssum))/abssum,
#            '0.1':      tf.maximum(1e-1, abssum)/abssum,
#            '0.01':     tf.maximum(1e-2, abssum)/abssum , 
#            '0.001':    tf.maximum(1e-3, abssum)/abssum ,
#            '0.0001':   tf.maximum(1e-4, abssum)/abssum \
#        }[args['normalization minimum gradient magnitude']]
if not args['base normalization']=='meansum': raise Exception("Not yet impl")

G_ = [gr[1] for gr in grads]
f_in, f_lyr, f_out = [lambda x: {'2': tf.square(x),  'abs': tf.abs(x),\
        '1': x,  'log': tf.log(tf.abs(x)),  '0.5': tf.sqrt(tf.abs(x)),\
        'max':  tf.reduce_max(x),  'logsumexp': tf.reduce_logsumexp(x) }[n] \
        for n in args['normalization beta: powers']]

virtual_gradsize = f_out(tf.reduce_sum([f_lyr(tf.reduce_sum(f_in(g))) for g in G_]))

normscale=args['normalization alpha: scaling']
if not normscale[:4]=='1 / ':
    normalizer = {'1':1.0, '1e-1':1e-1, '1e-2':1e-2, '1e-3':1e-3,\
            '1e-4':1e-4, '3e-6':3e-6, '1e1':1e1, '3e2':3e2}[normscale] 
else:
    if '^2' in normscale: vgs = lambda y: tf.square(y)
    elif 'sqrt' in normscale: vgs = lambda y: tf.sqrt(y)
    elif 'log' in normscale: vgs = lambda y: tf.log(y)
    else: vgs = lambda y: y

    const_scaling = float(normscale.split(' ')[2])
    normalizer =  1.0 / (const_scaling * vgs(virtual_gradsize))


thresholded_norm = tf.reduce_max(tf.constant(args\
        ['normalization gamma: threshold']), normalizer) / normalizer

nrm_grads = [(tf.multiply(thresholded_norm, grad),var) for grad, var in grads]


#normalized_grads 
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

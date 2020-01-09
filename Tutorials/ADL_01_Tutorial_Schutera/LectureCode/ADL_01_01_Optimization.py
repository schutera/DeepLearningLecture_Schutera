'''
Displaying the basic structure of a neural network and the optimization process
python -V 3.6.7
Contact: mark.schutera@kit.edu
'''

# version 1.3.0
import tensorflow as tf
# version 1.13.3
import numpy as np

np.random.seed(2)

# Neural Network architecture
n_input = 3
n_output = 1
n_units = 1

# Training parameters
n_updates = 10000


# Define graph / network
weights = {
    'h1': tf.Variable(np.reshape([np.float32(2.0), np.float32(2.0), np.float32(2.0)], (3, 1))),
    # 'h1': tf.Variable(tf.random_normal([n_input, n_units])),
}

biases = {
    'b1': tf.Variable(np.reshape([np.float32(4.0)], (1, 1))),
    # 'b1': tf.Variable(tf.random_normal([n_units])),
}


def unit(x0, weights, biases):
    # unit / neuron structure
    layer_1 = tf.add(tf.matmul(tf.cast(x0, tf.float32), weights['h1']), biases['b1'])
    # activation function
    y_pred = tf.nn.relu(layer_1)
    return y_pred


# Input
x = tf.Variable(np.reshape([1.0, 1.0, 3.0], (1, 3)))
# predicted output
y_pred = unit(x, weights, biases)
# Expected output
y_gt = tf.Variable(np.reshape([10.0], (1, 1)))


cost = tf.losses.mean_squared_error(labels=y_gt, predictions=y_pred)

# 1. Stochastic gradient descent
# opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

# 2. Momentum optimizer
# opt = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.75)

# 3. Adaptive momentum optimizer
# opt = tf.train.AdamOptimizer(learning_rate=0.0001)

# 4. Adaptive momentum optimizer with adjusted initial learning rate
# opt = tf.train.AdamOptimizer(learning_rate=0.1)

train = opt.minimize(cost)


# Run graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(n_updates):
        sess.run(train)
        grad, variable = sess.run(opt.compute_gradients(cost))[0]

        print('Gradient', np.reshape(grad, 3))
        print('Weights:', np.reshape(sess.run(weights['h1']), 3))
        print('Biases:', sess.run(biases['b1']))
        print('Prediction', y_pred.eval())
        print('')


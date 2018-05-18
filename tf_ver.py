import tensorflow as tf

train_in = [
  [1., 1.], [1., 0.],
  [0., 1.], [0., 0.]
]
train_out = [
  [0.], [1.],
  [1.], [0.]
]

# w1 to set a random 2x2 -> 2 (0)
w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.zeros([2]))
# w2 to set a random 2x1 -> 1 (0)
w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))
# out1 to be train_in X w1 + b1 ~ prediction
out1 = tf.nn.relu(tf.matmul(train_in, w1) + b1)
# out2 to be prediction X w2 + b2 
out2 = tf.nn.relu(tf.matmul(out1, w2) + b2)
# calculate error
error = tf.subtract(train_out, out2)
mse = tf.reduce_mean(tf.square(error))

train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

init = tf.global_variables_initializer()
sess = tf.Session()

err = 1.0
target = 0.01
epoch = 0
max_epochs = 1000

for i in range(max_epochs):
  sess.run(init, feed_dict={x_: train_in, y_: train_out})
  
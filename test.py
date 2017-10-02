import tensorflow as tf
import numpy as np

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
session = tf.Session()
var = tf.Variable([0.3], dtype=tf.float32)
var2 = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

init = tf.global_variables_initializer()
session.run(init)
adder_node = a+b
add_triple = adder_node*3
linear_model = var * x+var2
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
x_data = [1,2,3,4]
y_data = [0,-1,-2,-3]

print(session.run(adder_node, {a: 3.0, b: 4.5}))
print(session.run(adder_node, {a: [5.0, 3.0], b: [2.0, 4.0]}))
print(session.run(add_triple, {a: 3.0, b: 4.5}))
print(session.run(linear_model, {x: x_data}))
print(session.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


fixvar = tf.assign(var, [-1.0])
fixvar2 = tf.assign(var2, [1.0])

session.run([fixvar, fixvar2])
print(session.run(loss, {x: x_data, y: y_data}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
session.run(init)
for i in range(1000):
    session.run(train, {x: x_data, y: y_data})

print(session.run([var, var2]))


#curr_var = session.run(var, {x: x_data, y: y_data})
#curr_var2 = session.run(var2, {x: x_data, y: y_data})
#curr_loss = session.run(loss, {x: x_data, y: y_data})
#Same code, less lines. Worth it
curr_var, curr_var2, curr_loss = session.run([var, var2, loss], {x: x_data, y: y_data})

print("var: %s var2: %s loss: %s"%(curr_var, curr_var2, curr_loss))


# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)


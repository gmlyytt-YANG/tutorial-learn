from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def save_model():
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, [None, 784], name="myInput")
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b, name="myOutput")
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 1))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

    tf.saved_model.simple_save(sess, "./model", inputs={"myInput": x}, outputs={"myOutput": y})


def load_model():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./model")
        x = sess.graph.get_tensor_by_name('myInput:0')
        y = sess.graph.get_tensor_by_name('myOutput:0')
        batch_xs, batch_ys = mnist.test.next_batch(1)
        scores = sess.run(y, feed_dict={x: batch_xs})
        print("predict: %d, actual: %d" % (np.argmax(scores, 1), np.argmax(batch_ys, 1)))


load_model()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def weight_variable(shape, mean=0.0, stddev=0.1):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)

def bias_variable(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)

def conv_2d(x, W, stride=1):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, ksize=2):
    return tf.nn.max_pool(value=x, ksize=[1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')

def plot_2D(x, y, fig, subplot_idx, plot_info):
    ax = fig.add_subplot(subplot_idx)
    ax.plot(x, y)
    ax.set_xlabel(plot_info['xlabel'])
    ax.set_ylabel(plot_info['ylabel'])
    ax.set_title(plot_info['title'])

class fc(object):
    def __init__(self):
        # load data
        self.img = np.load('./datasets/random/random_imgs.npy') # 64*4*4
        self.label = np.load('./datasets/random/random_labs.npy')

        self.input = self.img.reshape(self.img.shape[0], -1)
        self.output = self.label.reshape(-1,1)

        # NN
        self.input_units = self.input.shape[-1]  # 4x4 pixel images
        self.hidden_units = 4  # 4-channel hidden layer
        self.output_units = 1  # single class
        self.learning_rate = 0.1
        self.num_epochs = 10000
        self.batch_size = 64  # training batch size

    def q1(self):
        # Use Sigmoid function as nueron activation and L2 loss for the network

        # parameters
        x = tf.placeholder(tf.float32, [None, self.input_units])
        y = tf.placeholder(tf.float32, [None, self.output_units])
        weights = {
            'hidden': weight_variable([self.input_units, self.hidden_units]),
            'output': weight_variable([self.hidden_units, self.output_units])
        }
        biases = {
            'hidden': bias_variable([self.hidden_units]),
            'output': bias_variable([self.output_units])
        }

        # graph
        layer1 = tf.matmul(x,weights['hidden'])+biases['hidden']
        activation1 = tf.nn.sigmoid(layer1)
        layer2 = tf.matmul(activation1, weights['output']) + biases['output']
        activation2 = tf.nn.sigmoid(layer2)

        # loss and acc
        loss = tf.reduce_mean(tf.square(activation2 - y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation2), y), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_history = []
            acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x:self.input, y:self.output})
                temp_loss, temp_acc = sess.run([loss, accuracy], {x: self.input, y: self.output})
                print ('Epoch ', epoch+1, '\tLoss : ', temp_loss, '\tAccuracy : ', temp_acc)
                iter_history.append(epoch)
                loss_history.append(temp_loss)
                acc_history.append(temp_acc)
                if temp_acc==1:
                    print('100% accuracy after', epoch+1, "iterations")
                    break

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(iter_history, loss_history, fig1, '121', {'xlabel': '#Iteration', 'ylabel': 'Loss', 'title': 'Sigmoid activation - L2 loss'})
        plot_2D(iter_history, acc_history, fig1, '122', {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'Sigmoid activation - Accuracy'})
        plt.show()

    def q2(self):
        # Use Sigmoid function as nueron activation and cross-entropy loss

        # parameters
        x = tf.placeholder(tf.float32, [None, self.input_units])
        y = tf.placeholder(tf.float32, [None, self.output_units])
        weights = {
            'hidden': weight_variable([self.input_units, self.hidden_units]),
            'output': weight_variable([self.hidden_units, self.output_units])
        }
        biases = {
            'hidden': bias_variable([self.hidden_units]),
            'output': bias_variable([self.output_units])
        }

        # graph
        layer1 = tf.matmul(x, weights['hidden']) + biases['hidden']
        activation1 = tf.nn.sigmoid(layer1)
        layer2 = tf.matmul(activation1, weights['output']) + biases['output']
        activation2 = tf.nn.sigmoid(layer2)

        # loss and acc
        loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(activation2,0,1)) + (1-y)*tf.log(tf.clip_by_value(1-activation2,0,1)))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation2), y), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_history = []
            acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x: self.input, y: self.output})
                temp_loss, temp_acc = sess.run([loss, accuracy], {x: self.input, y: self.output})
                print('Epoch ', epoch + 1, '\tLoss : ', temp_loss, '\tAccuracy : ', temp_acc)
                iter_history.append(epoch)
                loss_history.append(temp_loss)
                acc_history.append(temp_acc)
                if temp_acc == 1:
                    print('100% accuracy after', epoch + 1, "iterations")
                    break

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(iter_history, loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Loss', 'title': 'Sigmoid activation - CE loss'})
        plot_2D(iter_history, acc_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'Sigmoid activation - CE Accuracy'})
        plt.show()

    def q3(self):
        # Use ReLU activation and L2 loss
        # parameters
        x = tf.placeholder(tf.float32, [None, self.input_units])
        y = tf.placeholder(tf.float32, [None, self.output_units])
        weights = {
            'hidden': weight_variable([self.input_units, self.hidden_units]),
            'output': weight_variable([self.hidden_units, self.output_units])
        }
        biases = {
            'hidden': bias_variable([self.hidden_units]),
            'output': bias_variable([self.output_units])
        }

        # graph
        layer1 = tf.matmul(x, weights['hidden']) + biases['hidden']
        activation1 = tf.nn.relu(layer1)
        layer2 = tf.matmul(activation1, weights['output']) + biases['output']
        activation2 = tf.nn.sigmoid(layer2)

        # loss and acc
        loss = tf.reduce_mean(tf.square(activation2 - y))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation2), y), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_history = []
            acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x: self.input, y: self.output})
                temp_loss, temp_acc = sess.run([loss, accuracy], {x: self.input, y: self.output})
                print('Epoch ', epoch + 1, '\tLoss : ', temp_loss, '\tAccuracy : ', temp_acc)
                iter_history.append(epoch)
                loss_history.append(temp_loss)
                acc_history.append(temp_acc)
                if temp_acc == 1:
                    print('100% accuracy after', epoch + 1, "iterations")
                    break

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(iter_history, loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Loss', 'title': 'RELU activation - L2 loss'})
        plot_2D(iter_history, acc_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'RELU activation - L2 Accuracy'})
        plt.show()

    def q4(self):
        # Use ReLU activation and cross-entropy loss
        # parameters
        x = tf.placeholder(tf.float32, [None, self.input_units])
        y = tf.placeholder(tf.float32, [None, self.output_units])
        weights = {
            'hidden': weight_variable([self.input_units, self.hidden_units]),
            'output': weight_variable([self.hidden_units, self.output_units])
        }
        biases = {
            'hidden': bias_variable([self.hidden_units]),
            'output': bias_variable([self.output_units])
        }

        # graph
        layer1 = tf.matmul(x, weights['hidden']) + biases['hidden']
        activation1 = tf.nn.relu(layer1)
        layer2 = tf.matmul(activation1, weights['output']) + biases['output']
        activation2 = tf.nn.sigmoid(layer2)

        # loss and acc
        loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(activation2,0,1)) + (1-y)*tf.log(tf.clip_by_value(1-activation2,0,1)))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation2), y), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_history = []
            acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x: self.input, y: self.output})
                temp_loss, temp_acc = sess.run([loss, accuracy], {x: self.input, y: self.output})
                print('Epoch ', epoch + 1, '\tLoss : ', temp_loss, '\tAccuracy : ', temp_acc)
                iter_history.append(epoch)
                loss_history.append(temp_loss)
                acc_history.append(temp_acc)
                if temp_acc == 1:
                    print('100% accuracy after', epoch + 1, "iterations")
                    break

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(iter_history, loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Loss', 'title': 'RELU activation - CE loss'})
        plot_2D(iter_history, acc_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'RELU activation - CE Accuracy'})
        plt.show()



def main():
    p2 = fc()
    p2.q1()
    p2.q2()
    p2.q3()
    p2.q4()

if __name__ =='__main__':
    main()
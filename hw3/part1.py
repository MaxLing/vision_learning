import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from layers import *

class ConvNet():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 100
        self.max_iter = 2000
        self.test_iter = 100
        self.weight_decay = 0.5
        self.lr_start = 1e-3
        self.lr_decay = np.exp(np.log(1e-4/1e-3)/self.max_iter)
        c_num = 10

        '''build model'''
        # input
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3])
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.is_train = tf.placeholder(tf.bool)

        # layers
        conv1 = conv_factory(self.x, 32, [5, 5], 1, 2, self.is_train)
        conv2 = conv_factory(conv1, 64, [5, 5], 1, 2, self.is_train)
        conv3 = conv_factory(conv2, 128, [5, 5], 1, 2, self.is_train)
        conv4 = conv_factory(conv3, 256, [5, 5], 1, 2, self.is_train)
        conv5 = conv_factory(conv4, 512, [3, 3], 1, 2, self.is_train)

        flat = tf.layers.flatten(conv5)  # flatten for softmax
        dense = tf.layers.dense(flat, c_num)

        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=tf.one_hot(self.y, c_num)))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(dense, axis=1, output_type=tf.int32), self.y)))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, 1, self.lr_decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)

        with tf.control_dependencies([self.train]):
            self.wd_op = tf.train.GradientDescentOptimizer(self.lr).minimize(wd_loss)

    def run(self, imgs_train, labs_train, imgs_test, labs_test):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_loss_history = []
            train_acc_history = []
            test_acc_history = []
            for iter in range(self.max_iter):
                imgs_batch, labs_batch = sess.run([imgs_train, labs_train])
                _, _, train_loss, train_acc = sess.run([self.train, self.wd_op, self.loss, self.accuracy],
                                                    {self.x: imgs_batch, self.y: labs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss, '\t Train Accuracy : ', train_acc)
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                if (iter+1)%20==0: # run a full test
                    test_acc = 0
                    for i in range(self.test_iter):
                        imgs_batch, labs_batch = sess.run([imgs_test, labs_test])
                        test_acc += sess.run(self.accuracy, {self.x: imgs_batch, self.y: labs_batch, self.is_train: False})

                    test_acc /= self.test_iter
                    test_acc_history.append(test_acc)
                    print('Test Accuracy : ', test_acc)
                    if test_acc == 1:
                        print('100% accuracy after', iter + 1, "iterations")
                        break

            coord.request_stop()
            coord.join(threads)

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(train_loss_history, fig1, '131',
                {'xlabel': '#Iteration', 'ylabel': 'Train Loss', 'title': 'ConvNet'})
        plot_2D(train_acc_history, fig1, '132',
                {'xlabel': '#Iteration', 'ylabel': 'Train Accuracy', 'title': 'ConvNet'})
        plot_2D(test_acc_history, fig1, '133',
                {'xlabel': '#Iteration', 'ylabel': 'Test Accuracy', 'title': 'ConvNet'})
        plt.show()

class MobileNet():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 100
        self.max_iter = 2000
        self.test_iter = 100
        self.weight_decay = 0.5
        self.lr_start = 1e-3
        self.lr_decay = np.exp(np.log(1e-4/1e-3)/self.max_iter)
        c_num = 10

        '''build model'''
        # input
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3])
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.is_train = tf.placeholder(tf.bool)

        # layers
        conv1 = mobile_conv_factory(self.x, 32, [5, 5], 1, 2, self.is_train)
        conv2 = mobile_conv_factory(conv1, 64, [5, 5], 1, 2, self.is_train)
        conv3 = mobile_conv_factory(conv2, 128, [5, 5], 1, 2, self.is_train)
        conv4 = mobile_conv_factory(conv3, 256, [5, 5], 1, 2, self.is_train)
        conv5 = conv_factory(conv4, 512, [3, 3], 1, 2, self.is_train)

        flat = tf.layers.flatten(conv5)  # flatten for softmax
        dense = tf.layers.dense(flat, c_num)

        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=tf.one_hot(self.y, c_num)))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(dense, axis=1, output_type=tf.int32), self.y)))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, 1, self.lr_decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)

        with tf.control_dependencies([self.train]):
            self.wd_op = tf.train.GradientDescentOptimizer(self.lr).minimize(wd_loss)

    def run(self, imgs_train, labs_train, imgs_test, labs_test):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_loss_history = []
            train_acc_history = []
            test_acc_history = []
            for iter in range(self.max_iter):
                imgs_batch, labs_batch = sess.run([imgs_train, labs_train])
                _, train_loss, train_acc = sess.run([self.train, self.loss, self.accuracy],
                                                    {self.x: imgs_batch, self.y: labs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss, '\t Train Accuracy : ', train_acc)
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                if (iter+1)%20==0: # run a full test
                    test_acc = 0
                    for i in range(self.test_iter):
                        imgs_batch, labs_batch = sess.run([imgs_test, labs_test])
                        test_acc += sess.run(self.accuracy, {self.x: imgs_batch, self.y: labs_batch, self.is_train: False})

                    test_acc /= self.test_iter
                    test_acc_history.append(test_acc)
                    print('Test Accuracy : ', test_acc)
                    if test_acc == 1:
                        print('100% accuracy after', iter + 1, "iterations")
                        break

            coord.request_stop()
            coord.join(threads)

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(train_loss_history, fig1, '131',
                {'xlabel': '#Iteration', 'ylabel': 'Train Loss', 'title': 'MobileNet'})
        plot_2D(train_acc_history, fig1, '132',
                {'xlabel': '#Iteration', 'ylabel': 'Train Accuracy', 'title': 'MobileNet'})
        plot_2D(test_acc_history, fig1, '133',
                {'xlabel': '#Iteration', 'ylabel': 'Test Accuracy', 'title': 'MobileNet'})
        plt.show()

class ResNet():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 100
        self.max_iter = 2000
        self.test_iter = 100
        self.weight_decay = 0.5
        self.lr_start = 1e-3
        self.lr_decay = np.exp(np.log(1e-4/1e-3)/self.max_iter)
        c_num = 10

        '''build model'''
        # input
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3])
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.is_train = tf.placeholder(tf.bool)

        # layers
        conv1 = residual_block_factory(self.x, 32, [5, 5], 1, 2, self.is_train)
        conv2 = residual_block_factory(conv1, 64, [5, 5], 1, 2, self.is_train)
        conv3 = residual_block_factory(conv2, 128, [5, 5], 1, 2, self.is_train)
        conv4 = residual_block_factory(conv3, 256, [5, 5], 1, 2, self.is_train)
        conv5 = conv_factory(conv4, 512, [3, 3], 1, 2, self.is_train)

        flat = tf.layers.flatten(conv5)  # flatten for softmax
        dense = tf.layers.dense(flat, c_num)

        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=tf.one_hot(self.y, c_num)))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(dense, axis=1, output_type=tf.int32), self.y)))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, 1, self.lr_decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train = self.optimizer.minimize(self.loss)

        with tf.control_dependencies([self.train]):
            self.wd_op = tf.train.GradientDescentOptimizer(self.lr).minimize(wd_loss)

    def run(self, imgs_train, labs_train, imgs_test, labs_test):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_loss_history = []
            train_acc_history = []
            test_acc_history = []
            for iter in range(self.max_iter):
                imgs_batch, labs_batch = sess.run([imgs_train, labs_train])
                _, train_loss, train_acc = sess.run([self.train, self.loss, self.accuracy],
                                                    {self.x: imgs_batch, self.y: labs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss, '\t Train Accuracy : ', train_acc)
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                if (iter+1)%20==0: # run a full test
                    test_acc = 0
                    for i in range(self.test_iter):
                        imgs_batch, labs_batch = sess.run([imgs_test, labs_test])
                        test_acc += sess.run(self.accuracy, {self.x: imgs_batch, self.y: labs_batch, self.is_train: False})

                    test_acc /= self.test_iter
                    test_acc_history.append(test_acc)
                    print('Test Accuracy : ', test_acc)
                    if test_acc == 1:
                        print('100% accuracy after', iter + 1, "iterations")
                        break

            coord.request_stop()
            coord.join(threads)

        # plot
        fig1 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(train_loss_history, fig1, '131',
                {'xlabel': '#Iteration', 'ylabel': 'Train Loss', 'title': 'ResNet'})
        plot_2D(train_acc_history, fig1, '132',
                {'xlabel': '#Iteration', 'ylabel': 'Train Accuracy', 'title': 'ResNet'})
        plot_2D(test_acc_history, fig1, '133',
                {'xlabel': '#Iteration', 'ylabel': 'Test Accuracy', 'title': 'ResNet'})
        plt.show()

def plot_2D(y, fig, subplot_idx, plot_info):
	ax = fig.add_subplot(subplot_idx)
	ax.plot(np.arange(len(y)), y)
	ax.set_xlabel(plot_info['xlabel'])
	ax.set_ylabel(plot_info['ylabel'])
	ax.set_title(plot_info['title'])

def load_data_batch(root, split=None):
    filename = root + '/' + split

    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    labs_raw = dict[b'labels'] # 10000 images
    imgs_raw = dict[b'data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])

    with tf.device('/cpu:0'):
        imgs = tf.convert_to_tensor(imgs_raw, dtype=tf.float64)
        labs = tf.convert_to_tensor(labs_raw, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([imgs, labs], shuffle=True, capacity=1000)

        img = input_queue[0]
        lab = input_queue[1]
        img.set_shape([32, 32, 3])
        # img = tf.cast(img, tf.float32)
        img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1,
                                              batch_size=100, capacity=1000)
        '''batch can not be passed to graph directly, either sess.run to get value, or use in graph'''

    return img_batch, lab_batch

def main():
    # parameters
    root = './cifar10'
    train_split = 'data_batch_1'
    test_split = 'test_batch'

    # load data
    imgs_train, labs_train = load_data_batch(root, train_split)
    imgs_test, labs_test = load_data_batch(root, test_split)

    # model = ConvNet()
    model = MobileNet()
    # model = ResNet()
    model.run(imgs_train, labs_train, imgs_test, labs_test)

if __name__ =='__main__':
    main()
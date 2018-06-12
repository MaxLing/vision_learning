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

class cnn(object):
    def __init__(self):
        # NN
        self.conv1_kernel = 7
        self.conv1_size = 16
        self.conv2_kernel = 7
        self.conv2_size = 8
        self.output_units = 1  # single class
        self.learning_rate = 0.1
        self.num_epochs = 10000
        self.batch_size = 64

    def q1(self):
        # load data
        image = np.load('./datasets/line/line_imgs.npy')
        image_flat = image.reshape([-1, image.shape[1]*image.shape[2]])
        label = np.load('./datasets/line/line_labs.npy').reshape(-1,1)

        # input and output
        x = tf.placeholder(tf.float32, shape=[self.batch_size, image.shape[1]*image.shape[2]])
        x_reshaped = tf.reshape(x, [-1, image.shape[1], image.shape[2], 1])
        y = tf.placeholder(tf.float32, [self.batch_size, self.output_units])

        # graph
        conv1_w = weight_variable([self.conv1_kernel, self.conv1_kernel, 1, self.conv1_size])
        conv1_b = bias_variable([self.conv1_size])

        conv2_w = weight_variable([self.conv2_kernel, self.conv2_kernel, self.conv1_size, self.conv2_size])
        conv2_b = bias_variable([self.conv2_size])

        fc1_w = weight_variable([image.shape[1]*image.shape[2]*self.conv2_size,1])
        fc1_b = bias_variable([self.output_units])

        conv1 = conv_2d(x_reshaped, conv1_w) + conv1_b
        activation1 = tf.nn.relu(conv1)
        conv2 = conv_2d(activation1, conv2_w) + conv2_b
        activation2 = tf.nn.relu(conv2)
        activation2_flat = tf.reshape(activation2, [-1, image.shape[1]*image.shape[2]*self.conv2_size])
        fc1 = tf.matmul(activation2_flat, fc1_w) + fc1_b
        activation3 = tf.nn.sigmoid(fc1)

        # loss and acc
        loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(activation3,1e-10,1)) + (1-y)*tf.log(tf.clip_by_value(1-activation3,1e-10,1)))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation3), y), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            loss_history = []
            acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x: image_flat, y: label})
                temp_loss, temp_acc = sess.run([loss, accuracy], {x: image_flat, y: label})
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
                {'xlabel': '#Iteration', 'ylabel': 'Loss', 'title': 'RELU-CE loss'})
        plot_2D(iter_history, acc_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'RELU-CE Accuracy'})
        plt.show()

    def q2(self):
        # load data
        image = np.load('./datasets/detection/detection_imgs.npy')
        image_flat = image.reshape([-1, image.shape[1] * image.shape[2]])
        label = np.load('./datasets/detection/detection_labs.npy').reshape(-1, 1)
        width = np.load('./datasets/detection/detection_width.npy').reshape(-1, 1)

        # input and output
        x = tf.placeholder(tf.float32, shape=[self.batch_size, image.shape[1] * image.shape[2]])
        x_reshaped = tf.reshape(x, [-1, image.shape[1], image.shape[2], 1])
        y = tf.placeholder(tf.float32, [self.batch_size, self.output_units])
        w = tf.placeholder(tf.float32, [self.batch_size, 1])

        # graph
        conv1_w = weight_variable([self.conv1_kernel, self.conv1_kernel, 1, self.conv1_size])
        conv1_b = bias_variable([self.conv1_size])

        conv2_w = weight_variable([self.conv2_kernel, self.conv2_kernel, self.conv1_size, self.conv2_size])
        conv2_b = bias_variable([self.conv2_size])

        fc1_w = weight_variable([image.shape[1] * image.shape[2] * self.conv2_size, 1])
        fc1_b = bias_variable([self.output_units])
        fc2_w = weight_variable([image.shape[1] * image.shape[2] * self.conv2_size, 1])
        fc2_b = bias_variable([self.output_units])

        conv1 = conv_2d(x_reshaped, conv1_w) + conv1_b
        activation1 = tf.nn.relu(conv1)
        conv2 = conv_2d(activation1, conv2_w) + conv2_b
        activation2 = tf.nn.relu(conv2)
        activation2_flat = tf.reshape(activation2, [-1, image.shape[1] * image.shape[2] * self.conv2_size])
        fc1 = tf.matmul(activation2_flat, fc1_w) + fc1_b
        activation3 = tf.nn.sigmoid(fc1)
        fc2 = tf.matmul(activation2_flat, fc2_w) + fc2_b

        # loss and acc
        class_loss = -tf.reduce_mean(y * tf.log(tf.clip_by_value(activation3, 1e-10, 1)) + (1 - y) * tf.log(tf.clip_by_value(1 - activation3, 1e-10, 1)))
        class_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(activation3), y), tf.float32))
        reg_loss = tf.reduce_mean(tf.square(fc2 - w))
        reg_acc = tf.reduce_mean(tf.cast(tf.less(tf.abs(fc2 - w), 0.5), tf.float32))

        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        loss = class_loss + 0.01 * reg_loss
        train = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            class_loss_history = []
            class_acc_history = []
            reg_loss_history = []
            reg_acc_history = []
            iter_history = []
            for epoch in range(self.num_epochs):
                sess.run(train, {x: image_flat, y: label, w:width})
                temp_class_loss, temp_class_acc, temp_reg_loss, temp_reg_acc\
                    = sess.run([class_loss, class_acc, reg_loss, reg_acc], {x: image_flat, y: label, w:width})
                print('Epoch ', epoch + 1, '\tClassification Loss : ', temp_class_loss, '\t Classification Accuracy : ', temp_class_acc,
                      '\tRegression Loss : ', temp_reg_loss, '\t Regression Accuracy : ', temp_reg_acc)
                iter_history.append(epoch)
                class_loss_history.append(temp_class_loss)
                class_acc_history.append(temp_class_acc)
                reg_loss_history.append(temp_reg_loss)
                reg_acc_history.append(temp_reg_acc)
                if temp_reg_acc == 1:
                    print('100% regression accuracy after', epoch + 1, "iterations")
                    break

        # plot
        fig1 = plt.figure()
        plot_2D(iter_history, class_loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Classification Loss', 'title': 'RELU-CE loss'})
        plot_2D(iter_history, class_acc_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Accuracy', 'title': 'RELU-CE Accuracy'})
        plt.show()

        fig2 = plt.figure()
        plot_2D(iter_history, reg_loss_history, fig2, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Regression Loss', 'title': 'RELU-L2 loss'})
        plot_2D(iter_history, reg_acc_history, fig2, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Regression Accuracy', 'title': 'RELU-L2 Accuracy'})
        plt.show()

def main():
    p3 = cnn()
    # p3.q1()
    p3.q2()

if __name__ =="__main__":
    main()
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import *

class FasterRCNN():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 100
        self.epoch_size = int(10000/self.batch_size)
        self.max_iter = 2000
        self.weight_decay = 0.05
        self.lr_start = 1e-3
        self.lr_decay = (1e-4/1e-3)**(1./(self.max_iter/self.epoch_size-1))
        c_num = 10

        '''build model'''
        # input
        self.imgs = tf.placeholder(tf.float32, shape=[self.batch_size, 48, 48, 3])
        self.labs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.msks = tf.placeholder(tf.int32, shape=[self.batch_size, 6, 6, 1]) # downsample by 8, 3 times pooling
        self.xs = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.ys = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.ws = tf.placeholder(tf.float32, shape=[self.batch_size])
        self.is_train = tf.placeholder(tf.bool)

        # base net
        conv1 = conv_factory(self.imgs, 32, [5, 5], 1, 2, self.is_train)
        conv2 = conv_factory(conv1, 64, [5, 5], 1, 2, self.is_train)
        conv3 = conv_factory(conv2, 128, [5, 5], 1, 2, self.is_train)
        conv4 = conv_factory(conv3, 256, [5, 5], 1, 2, self.is_train, pooling=False)
        self.feat = conv4
        inter = conv_factory(self.feat, 256, [3, 3], 1, 2, self.is_train, pooling=False)

        # proposal cls
        with tf.variable_scope('cls', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weights', [1, 1, 256, 1],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable('biases', [1, 1, 1, 1],
                                initializer=tf.constant_initializer(0.0))
            cls = tf.nn.conv2d(inter, w, strides=[1, 1, 1, 1], padding='SAME') + b

            loss_mask = tf.not_equal(self.msks, 2)[:, :, :, 0] # skip white region
            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls, labels=tf.cast(tf.equal(self.msks, 1), tf.float32))
            self.cls_loss = tf.reduce_mean(tf.boolean_mask(cls_loss, loss_mask))

            cls_output = tf.cast(tf.round(tf.nn.sigmoid(cls)), tf.int32)
            cls_acc = tf.to_float(tf.equal(cls_output, self.msks))
            self.cls_acc = tf.reduce_mean(tf.boolean_mask(cls_acc, loss_mask))

        # proposal reg



        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        wd_loss = tf.add_n(tf.get_collection('losses'))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, self.epoch_size, self.lr_decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.wd_op = tf.train.GradientDescentOptimizer(self.lr).minimize(wd_loss)

    def train_cls(self, imgs_train, masks_train, imgs_test, masks_test):
        self.train = self.optimizer.minimize(self.cls_loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_loss_history = []
            train_acc_history = []
            test_acc_history = []
            for iter in range(self.max_iter):
                imgs_batch, labs_batch = sess.run([imgs_train, masks_train])
                _, _, train_loss, train_acc = sess.run([self.train, self.wd_op, self.cls_loss, self.cls_acc],
                                                    {self.imgs: imgs_batch, self.msks: labs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss, '\t Train Accuracy : ', train_acc)
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                if (iter+1)%self.epoch_size==0: # run a full test
                    test_acc = 0
                    for i in range(self.epoch_size):
                        imgs_batch, labs_batch = sess.run([imgs_test, masks_test])
                        test_acc += sess.run(self.cls_acc, {self.imgs: imgs_batch, self.msks: labs_batch, self.is_train: False})

                    test_acc /= self.epoch_size
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
                {'xlabel': '#Iteration', 'ylabel': 'Train Loss', 'title': 'Proposal Classifier'})
        plot_2D(train_acc_history, fig1, '132',
                {'xlabel': '#Iteration', 'ylabel': 'Train Accuracy', 'title': 'Proposal Classifier'})
        plot_2D(test_acc_history, fig1, '133',
                {'xlabel': '#Iteration', 'ylabel': 'Test Accuracy', 'title': 'Proposal Classifier'})
        plt.show()

def plot_2D(y, fig, subplot_idx, plot_info):
	ax = fig.add_subplot(subplot_idx)
	ax.plot(np.arange(len(y)), y)
	ax.set_xlabel(plot_info['xlabel'])
	ax.set_ylabel(plot_info['ylabel'])
	ax.set_title(plot_info['title'])

def read_from_file(file, image_path, mask_path):
    imgs = []
    labs = []
    masks = []
    xs = []
    ys = []
    ws = []
    with open(file, 'r') as f:
        for line in f:
            image_name, label, x, y, w = line[:-1].split(' ')
            imgs.append(image_path + image_name)
            masks.append(mask_path + image_name)
            labs.append(int(label))
            xs.append(int(x))
            ys.append(int(y))
            ws.append(int(w))
    return imgs, labs, masks, xs, ys, ws

def read_from_disk(input_queue):
    img_path = tf.read_file(input_queue[0])
    img = tf.image.decode_png(img_path, channels=3)
    mask_path = tf.read_file(input_queue[2])
    mask = tf.image.decode_png(mask_path, channels=1)

    lab = input_queue[1]
    x = input_queue[3]
    y = input_queue[4]
    w = input_queue[5]

    return img, lab, mask, x, y, w

def load_data_batch(root, split):
    image_path = root+'/imgs/'
    mask_path = root+'/masks/'
    file = root+'/devkit/'+split+'.txt'
    imgs_raw, labs_raws, masks_raw, x_raw, y_raw, w_raw = read_from_file(file, image_path, mask_path) # note img, mask are path

    with tf.device('/cpu:0'):
        imgs = tf.convert_to_tensor(imgs_raw, dtype=tf.string)
        labs = tf.convert_to_tensor(labs_raws, dtype=tf.int32)
        masks = tf.convert_to_tensor(masks_raw, dtype=tf.string)
        xs = tf.convert_to_tensor(x_raw, dtype=tf.int32)
        ys = tf.convert_to_tensor(y_raw, dtype=tf.int32)
        ws = tf.convert_to_tensor(w_raw, dtype=tf.int32)

        input_queue = tf.train.slice_input_producer([imgs, labs, masks, xs, ys, ws], shuffle=True, capacity=1000)

        img, lab, mask, x, y, w = read_from_disk(input_queue)
        img.set_shape([48, 48, 3])
        mask.set_shape([6, 6, 1])

        img_batch, lab_batch, mask_batch, x_batch, y_batch, w_batch = \
            tf.train.batch([img, lab, mask, x, y, w], num_threads=1, batch_size=100, capacity=10000)
        '''batch can not be passed to graph directly, either sess.run to get value, or use in graph'''

    return img_batch, lab_batch, mask_batch, x_batch, y_batch, w_batch

def main():
    # parameters
    root = './cifar10_transformed'

    # load data
    imgs_train, labs_train, masks_train, xs_train, ys_train, ws_train = load_data_batch(root, split='train')
    imgs_test, labs_test, masks_test, xs_test, ys_test, ws_test = load_data_batch(root, split='test')

    model = FasterRCNN()
    # task 1
    model.train_cls(imgs_train, masks_train, imgs_test, masks_test)
    # # task 2
    # model.train_rpn()
    

if __name__ =='__main__':
    main()
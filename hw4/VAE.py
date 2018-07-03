import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import *

class VAE():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 50
        self.epoch_size = 2
        self.max_iter = 3000
        self.weight_decay = 0.05
        self.lr_start = 1e-3
        self.lr_decay = (1e-4 / 1e-3) ** (1. / (self.max_iter / self.epoch_size - 1))

        '''build model'''
        # input
        self.imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
        self.is_train = tf.placeholder(tf.bool)

        # encoder 1(3)->4->128
        self.feat = encoder(self.imgs, 4, [3,3], 2, self.is_train, 6)

        # decoder 128->4->1(3) from decoder is without BN or activation
        gen = decoder(self.feat, 64, 1, [3,3], 2, self.is_train, 6)
        self.gen = tf.nn.sigmoid(gen)*255

        # loss
        self.loss = tf.reduce_mean(tf.pow(self.gen-self.imgs, 2))

        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        self.wd_loss = tf.add_n(tf.get_collection('losses'))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, self.epoch_size, self.lr_decay,
                                             staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)

    def train(self, imgs_train, labs_train, imgs_test, labs_test):
        # right way to use BN layer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = self.optimizer.minimize(self.loss)
            self.wd_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.wd_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            train_loss_history = []
            test_loss_history = []
            for iter in range(self.max_iter):
                imgs_batch = sess.run(imgs_train)
                _, _, train_loss, train_gen = sess.run([self.train, self.wd_op, self.loss, self.gen],
                                                    {self.imgs: imgs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss)
                train_loss_history.append(train_loss)

                if (iter+1)%200==0: # run a full test
                    test_loss = 0
                    for i in range(self.epoch_size):
                        imgs_batch = sess.run(imgs_test)
                        test_loss += sess.run(self.loss, {self.imgs: imgs_batch, self.is_train: False})

                    test_loss /= self.epoch_size
                    test_loss_history.append(test_loss)
                    print('Test Loss : ', test_loss)

                    # show training generated images
                    img1 = np.squeeze(train_gen[0,:,:,:])
                    img2 = np.squeeze(train_gen[-1,:,:,:])
                    plt.figure(1)
                    plt.subplot(211)
                    plt.imshow(img1, cmap='gray')
                    plt.subplot(212)
                    plt.imshow(img2, cmap='gray')
                    plt.title('training')
                    plt.show()

                if iter == self.max_iter-1:
                    # show testing generated images
                    imgs_batch = sess.run(imgs_test)
                    test_gen = sess.run(self.gen, {self.imgs: imgs_batch, self.is_train: False})

                    for j in range(10):
                        org_img = np.squeeze(imgs_batch[10*j,:,:,:])
                        test_img = np.squeeze(test_gen[10*j,:,:,:])
                        plt.figure(j)
                        plt.subplot(211)
                        plt.imshow(org_img, cmap='gray')
                        plt.subplot(212)
                        plt.imshow(test_img, cmap='gray')
                        plt.title('testing')
                        plt.show()

            coord.request_stop()
            coord.join(threads)

        # plot
        fig1 = plt.figure()
        plot_2D(train_loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'Train Loss', 'title': 'VAE'})
        plot_2D(test_loss_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'Test Loss', 'title': 'VAE'})
        plt.show()


def plot_2D(y, fig, subplot_idx, plot_info):
	ax = fig.add_subplot(subplot_idx)
	ax.plot(np.arange(len(y)), y)
	ax.set_xlabel(plot_info['xlabel'])
	ax.set_ylabel(plot_info['ylabel'])
	ax.set_title(plot_info['title'])

def read_from_file(file, image_path):
    imgs = []
    labs = []
    with open(file, 'r') as f:
        for line in f:
            image_name, label = line[:-1].split(' ')
            imgs.append(image_path + image_name)
            labs.append(int(label))
    return imgs, labs

def read_from_disk(input_queue):
    img_path = tf.read_file(input_queue[0])
    img = tf.image.decode_png(img_path, channels=1)

    lab = input_queue[1]

    return img, lab

def load_data_batch(root, split):
    image_path = root+'/imgs/'
    file = root+'/devkit/'+split+'.txt'
    imgs_raw, labs_raws = read_from_file(file, image_path)

    with tf.device('/cpu:0'):
        imgs = tf.convert_to_tensor(imgs_raw, dtype=tf.string)
        labs = tf.convert_to_tensor(labs_raws, dtype=tf.int32)

        input_queue = tf.train.slice_input_producer([imgs, labs], shuffle=True, capacity=1000)

        img, lab = read_from_disk(input_queue)
        img.set_shape([64, 64, 1])

        img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1, batch_size=100, capacity=10000, allow_smaller_final_batch=True)
        '''batch can not be passed to graph directly, either sess.run to get value, or use in graph'''

    return img_batch, lab_batch

def main():
    # parameters
    root = './cufs'

    # load data
    imgs_train, labs_train = load_data_batch(root, split='train')
    imgs_test, labs_test = load_data_batch(root, split='test')

    # model
    model = VAE()
    model.train(imgs_train, labs_train, imgs_test, labs_test)




if __name__ =='__main__':
    main()
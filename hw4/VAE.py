import numpy as np
import tensorflow as tf

from layers import *
from utils import *

class VAE():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 50
        self.epoch_size = 2
        self.max_iter = 2000
        self.weight_decay = 0.05
        self.lr_start = 1e-3
        self.lr_decay = (1e-4 / 1e-3) ** (1. / (self.max_iter / self.epoch_size - 1))
        self.latent_num = 64

        '''build model'''
        # input
        self.imgs = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
        self.is_train = tf.placeholder(tf.bool)

        # encoder 1(3)->4->128
        self.feature = encoder(self.imgs, 4, [3,3], 2, self.is_train, 6)
        latent_shape =self.feature.get_shape()

        with tf.variable_scope('latent', reuse=tf.AUTO_REUSE):
            feature = tf.layers.flatten(self.feature)
            mean = tf.layers.dense(feature, self.latent_num)
            std = tf.layers.dense(feature, self.latent_num)
            cov = tf.square(std)

            self.latent = tf.random_normal(tf.shape(std))*cov + mean
            self.KL_loss = 0.5*tf.reduce_mean(tf.square(mean)+cov-tf.log(cov)-1)

            latent = tf.layers.dense(self.latent, latent_shape[1]*latent_shape[2]*latent_shape[3])
            latent = tf.reshape(latent, tf.shape(self.feature))

        # decoder 128->4->1(3) from decoder is without BN or activation
        gen = decoder(latent, 64, 1, [3,3], 2, self.is_train, 6)
        self.generate = tf.nn.sigmoid(gen)

        # loss
        self.ML_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen, labels=self.imgs))

        '''training'''
        for var in tf.trainable_variables():
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
        self.wd_loss = tf.add_n(tf.get_collection('losses'))

        self.lr = tf.train.exponential_decay(self.lr_start, self.max_iter, self.epoch_size, self.lr_decay,
                                             staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.loss = self.KL_loss+self.ML_loss

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
                imgs_batch = sess.run(imgs_train)/255
                _, _, train_loss, train_gen = sess.run([self.train, self.wd_op, self.loss, self.generate],
                                                    {self.imgs: imgs_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss)
                train_loss_history.append(train_loss)

                if (iter+1)%100==0: # run a full test
                    test_loss = 0
                    for i in range(self.epoch_size):
                        imgs_batch = sess.run(imgs_test)/255
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

                if iter+1 == self.max_iter:
                    # show testing generated images
                    imgs_batch = sess.run(imgs_test)/255
                    test_gen = sess.run(self.generate, {self.imgs: imgs_batch, self.is_train: False})

                    for j in range(10):
                        org_img = np.squeeze(imgs_batch[5*j,:,:,:])
                        test_img = np.squeeze(test_gen[5*j,:,:,:])
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
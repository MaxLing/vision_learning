import numpy as np
import tensorflow as tf

from layers import *
from utils import *

class CGAN():
    def __init__(self):
        '''model parameters'''
        self.batch_size = 100
        self.epoch_size = 2
        self.max_iter = 2000
        self.lr = 1e-4

        self.latent_num = 100
        self.label_smooth = 0.9
        self.image_size = np.array([64,64,1])
        self.hierarchy = 4
        self.d_filter = 32
        self.g_filter = self.d_filter*(2**(self.hierarchy-2))
        self.latent_size = (self.image_size[:2]/(2**self.hierarchy)).astype(int)
        self.kernel_size = [5,5]
        self.conv_strides = 2

    def discriminator(self, sample, label, is_train):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # sample is real or generated images [64,64,2]->[4,4,128]
            # 1 more channel for label (conditional)
            label = tf.reshape(label, [-1, 1, 1, 1])
            label = tf.tile(label, [1, 64, 64, 1])
            sample += tf.random_normal(shape=tf.shape(sample), mean=0.0, stddev=0.1, dtype=tf.float32)
            sample = tf.concat([sample, label], axis=3)

            feature = encoder(sample, self.d_filter, self.kernel_size, self.conv_strides, is_train, self.hierarchy, activation='leaky_relu')
            output = tf.layers.dense(tf.layers.flatten(feature), 1)
            return output

    def generator(self, sample, label, is_train):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # sample is nd vector [4,4,128]->[64,64,1]
            # 1 more entry for label (conditional)
            label = tf.reshape(label, [-1, 1])
            sample = tf.reshape(sample, [-1, self.latent_num])
            sample = tf.concat([sample, label], axis=1)

            latent = tf.layers.dense(sample, self.latent_size[0]*self.latent_size[1]*self.g_filter)
            latent = tf.reshape(latent, [-1, self.latent_size[0], self.latent_size[1], self.g_filter])

            gen = decoder(latent, self.g_filter, self.image_size[2], self.kernel_size, self.conv_strides, is_train, self.hierarchy)
            return tf.nn.tanh(gen)

    def train(self, imgs_train, labs_train, imgs_test, labs_test):
        z = tf.placeholder(tf.float32, [self.batch_size, self.latent_num]) # [-1,1]

        true_labels = tf.concat([labs_train, labs_test], axis=0)
        true_imgs = tf.concat([imgs_train, imgs_test], axis=0)/255*2-1 # more imgs for training, normalize to [-1,1]

        fake_imgs = self.generator(z, true_labels, True)
        fake_score = self.discriminator(fake_imgs, true_labels, True)
        true_score = self.discriminator(true_imgs, true_labels, True)

        # 2 cross entropy loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=self.label_smooth*tf.ones_like(fake_score)))
        d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=true_score, labels=self.label_smooth*tf.ones_like(true_score))) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=(1-self.label_smooth)*tf.zeros_like(fake_score)))
        g_score = tf.reduce_mean(tf.nn.sigmoid(fake_score))
        d_score = tf.reduce_mean(tf.nn.sigmoid(true_score))

        # important to seperate
        g_var = [var for var in tf.trainable_variables() if 'generator' in var.name]
        d_var = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_op = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
                .minimize(g_loss, var_list=g_var)
            d_op = tf.train.AdamOptimizer(learning_rate=self.lr/5, beta1=0.5) \
                .minimize(d_loss, var_list=d_var)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            g_loss_history = []
            d_loss_history = []
            for iter in range(self.max_iter):
                _, gen_loss, dis_loss, dis_score, gen_score = sess.run([d_op, g_loss, d_loss, d_score, g_score], \
                                                {z: np.random.uniform(-1.,1.,size =[self.batch_size, self.latent_num])})
                _ = sess.run(g_op, {z: np.random.uniform(-1.,1.,size =[self.batch_size, self.latent_num])})

                print('Iter ', iter + 1, '\tG_Loss: ', gen_loss, '\tD_Loss: ', dis_loss, '\tD_Score: ', dis_score, '\tG_Score: ', gen_score)
                g_loss_history.append(gen_loss)
                d_loss_history.append(dis_loss)

                if (iter+1)%500==0: # show some generate
                    generate = sess.run(fake_imgs, {z: np.random.uniform(-1.,1.,size =[self.batch_size, self.latent_num])})
                    generate = (generate+1)/2 # to [0,1]

                    img1 = np.squeeze(generate[0,:,:,:])
                    img2 = np.squeeze(generate[-1,:,:,:])
                    plt.figure(1)
                    plt.subplot(211)
                    plt.imshow(img1, cmap='gray')
                    plt.subplot(212)
                    plt.imshow(img2, cmap='gray')
                    plt.show()

            coord.request_stop()
            coord.join(threads)
            tf.train.Saver().save(sess, './model/CGAN')

        # plot
        fig1 = plt.figure()
        plot_2D(g_loss_history, fig1, '121',
                {'xlabel': '#Iteration', 'ylabel': 'G Loss', 'title': 'GAN'})
        plot_2D(d_loss_history, fig1, '122',
                {'xlabel': '#Iteration', 'ylabel': 'D Loss', 'title': 'GAN'})
        plt.show()

    def generate(self, path):
        gen_num = 10
        z = tf.placeholder(tf.float32, [gen_num, self.latent_num])
        fake_labels = np.arange(gen_num, dtype=np.float32)
        fake_imgs = self.generator(z, fake_labels, False)
        g_score = tf.reduce_mean(tf.nn.sigmoid(self.discriminator(fake_imgs, fake_labels, False)))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver().restore(sess, path)

            generate, gen_score = sess.run([fake_imgs, g_score], \
                                           {z: np.random.uniform(-1., 1., size=[gen_num, self.latent_num])})
            generate = (generate+1)/2  # to [0,1]

        print(gen_score)
        for i in range(gen_num):
            plt.figure(i)
            plt.imshow(np.squeeze(generate[i,:,:,:]), cmap='gray')
            plt.show()



def main():
    # parameters
    root = './cufs'

    # load data
    imgs_train, labs_train = load_data_batch(root, split='train')
    imgs_test, labs_test = load_data_batch(root, split='test')

    # model
    model = CGAN()
    model.train(imgs_train, labs_train, imgs_test, labs_test)
    model.generate('./model/CGAN')

if __name__ =='__main__':
    main()
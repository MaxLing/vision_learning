import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from layers import *
from spatial_transformer import transformer

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
        inter = conv_factory(conv4, 256, [3, 3], 1, 2, self.is_train, pooling=False)

        # anchors with or without high IoU: foreground or background
        with tf.variable_scope('cls', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weights', [1, 1, 256, 1],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable('biases', [1, 1, 1, 1],
                                initializer=tf.constant_initializer(0.0))
            cls = tf.nn.conv2d(inter, w, strides=[1, 1, 1, 1], padding='SAME') + b
            max_cls = tf.argmax(tf.reshape(tf.nn.sigmoid(cls),[self.batch_size, -1, 1]), axis=1)

            loss_mask = tf.not_equal(self.msks, 2) # skip unclear region, not fg or bg
            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=cls, labels=tf.cast(tf.equal(self.msks, 1), tf.float32))
            self.cls_loss = tf.reduce_mean(tf.boolean_mask(cls_loss, loss_mask))

            self.cls_output = tf.cast(tf.round(tf.nn.sigmoid(cls)), tf.int32)
            cls_acc = tf.to_float(tf.equal(self.cls_output, self.msks))
            self.cls_acc = tf.reduce_mean(tf.boolean_mask(cls_acc, loss_mask))

        # from anchor to a nearby ground-truth box
        with tf.variable_scope('reg', reuse=tf.AUTO_REUSE):
            w = tf.get_variable('weights', [1, 1, 256, 3],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable('biases', [1, 1, 1, 3],
                                initializer=tf.constant_initializer([24, 24, 32]))
            reg = tf.nn.conv2d(inter, w, strides=[1, 1, 1, 1], padding='SAME') + b

            # loss_mask = tf.equal(self.msks, 1)[:, :, :, 0]
            loss_mask = tf.equal(self.cls_output, 1)[:, :, :, 0]

            def smooth_l1(x):
                return tf.where(tf.less(tf.abs(x), 1), 0.5 * tf.pow(x, 2), tf.abs(x) - 0.5)
            def enlarge(x):
                return tf.tile(tf.reshape(x, [100, 1, 1]), [1, 6, 6])

            # simplify equations in article
            tx_diff = (reg[:,:,:,0]-enlarge(self.xs))/32 # 32 is anchor width
            ty_diff = (reg[:,:,:,1]-enlarge(self.ys))/32
            tw_diff = tf.log(reg[:,:,:,2]/enlarge(self.ws))
            reg_loss = smooth_l1(tx_diff) + smooth_l1(ty_diff) + smooth_l1(tw_diff)
            self.reg_loss = tf.reduce_mean(tf.boolean_mask(reg_loss, loss_mask))

        with tf.variable_scope('fast-rcnn', reuse=tf.AUTO_REUSE):
            # RoI pooling
            # concat 100*2 index tensors
            x_reg = tf.reshape(reg, [self.batch_size, -1, 3])
            max_reg = tf.gather_nd(x_reg, tf.concat(
                [tf.expand_dims(tf.range(self.batch_size, dtype=tf.int64), 1), max_cls], axis=1))

            theta = tf.transpose(tf.convert_to_tensor(
                [max_reg[:, 2] / 48, tf.constant(0, tf.float32, shape=[self.batch_size]), (max_reg[:, 0] - 24) / 24,
                 tf.constant(0, tf.float32, shape=[self.batch_size]), max_reg[:, 2] / 48, (max_reg[:, 1] - 24) / 24]))
            self.imgs_cropped = transformer(self.imgs, theta, [32, 32])
            self.feat_cropped = transformer(self.feat, theta, [4, 4])
            self.imgs_cropped.set_shape([self.batch_size,32,32,3])
            self.feat_cropped.set_shape([self.batch_size,4,4,256])

            # object classification
            conv6 = conv_factory(self.feat_cropped, 256, [3, 3], 1, 2, self.is_train, pooling=False)
            flat = tf.layers.flatten(conv6)
            dense = tf.layers.dense(flat, c_num)

            self.obj_cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense, labels=tf.one_hot(self.labs, c_num)))
            self.obj_cls_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(dense, axis=1, output_type=tf.int32), self.labs)))

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
                imgs_batch, msks_batch = sess.run([imgs_train, masks_train])
                _, _, train_loss, train_acc = sess.run([self.train, self.wd_op, self.cls_loss, self.cls_acc],
                                                    {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: True})
                print('Iter ', iter + 1, '\tTrain Loss : ', train_loss, '\t Train Accuracy : ', train_acc)
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                if (iter+1)%self.epoch_size==0: # run a full test
                    test_acc = 0
                    for i in range(self.epoch_size):
                        imgs_batch, msks_batch = sess.run([imgs_test, masks_test])
                        test_acc += sess.run(self.cls_acc, {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: False})

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

    def train_rpn(self, imgs_train, masks_train, xs_train, ys_train, ws_train, imgs_test, masks_test, xs_test, ys_test, ws_test):
        loss = 100*(self.cls_loss + self.reg_loss)
        self.train = self.optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            cls_loss_history = []
            cls_acc_history = []
            reg_loss_history = []
            test_loss_history = []
            for iter in range(self.max_iter):
                imgs_batch, msks_batch, xs_batch, ys_batch, ws_batch = sess.run([imgs_train, masks_train, xs_train, ys_train, ws_train])
                _, _, cls_loss, cls_acc, reg_loss = sess.run([self.train, self.wd_op, self.cls_loss, self.cls_acc, self.reg_loss],
                                                    {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: True,
                                                     self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})
                print('Iter ', iter + 1, '\t Cls Loss : ', cls_loss, '\t Cls Accuracy : ', cls_acc, '\t Reg Loss: ', reg_loss)
                cls_loss_history.append(cls_loss)
                cls_acc_history.append(cls_acc)
                reg_loss_history.append(reg_loss)

                if (iter+1)%self.epoch_size==0: # run a full test
                    test_reg_loss = 0
                    test_cls_loss = 0
                    test_cls_acc = 0
                    for i in range(self.epoch_size):
                        imgs_batch, msks_batch, xs_batch, ws_batch, zs_batch = sess.run([imgs_test, masks_test, xs_test, ys_test, ws_test])
                        temp_cls_loss, temp_cls_acc, temp_reg_loss =  sess.run([self.cls_loss, self.cls_acc, self.reg_loss],
                                                    {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: False,
                                                     self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})

                        test_cls_loss += temp_cls_loss
                        test_cls_acc += temp_cls_acc
                        test_reg_loss += temp_reg_loss

                        if (i+1)%self.epoch_size==0:
                            imgs, imgs_cropped, proposals, masks = sess.run([self.imgs, self.imgs_cropped, self.cls_output, self.msks],
                                                          {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: False,
                                                           self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})
                            img = np.clip(np.squeeze(imgs[0,:,:,:])/256., 0, 1)
                            img_cropped = np.clip(np.squeeze(imgs_cropped[0,:,:,:])/256., 0, 1)
                            proposal = np.clip(np.squeeze(proposals[0,:,:,:])/2., 0, 1)
                            mask = np.clip(np.squeeze(masks[0,:,:,:])/2., 0, 1)

                            plt.figure(4)
                            plt.subplot(211)
                            plt.imshow(img)
                            plt.subplot(212)
                            plt.imshow(img_cropped)
                            plt.show()
                            plt.figure(5)
                            plt.subplot(211)
                            plt.imshow(proposal)
                            plt.subplot(212)
                            plt.imshow(mask)
                            plt.show()

                    test_cls_loss /= self.epoch_size
                    test_cls_acc /= self.epoch_size
                    test_reg_loss /= self.epoch_size
                    test_loss_history.append(test_reg_loss)
                    print('Test Cls Loss: ', test_cls_loss, '\tTest Cls Acc: ', test_cls_acc, '\tTest Reg Loss : ', test_reg_loss)

            coord.request_stop()
            coord.join(threads)

        # plot
        fig2 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(cls_loss_history, fig2, '141',
                {'xlabel': '#Iteration', 'ylabel': 'Train Cls Loss', 'title': 'RPN'})
        plot_2D(cls_acc_history, fig2, '142',
                {'xlabel': '#Iteration', 'ylabel': 'Train Cls Accuracy', 'title': 'RPN'})
        plot_2D(reg_loss_history, fig2, '143',
                {'xlabel': '#Iteration', 'ylabel': 'Train Reg Loss', 'title': 'RPN'})
        plot_2D(test_loss_history, fig2, '144',
                {'xlabel': '#Iteration', 'ylabel': 'Test Reg Loss', 'title': 'RPN'})
        plt.show()

    def train(self, imgs_train, labs_train, masks_train, xs_train, ys_train, ws_train,
              imgs_test, labs_test, masks_test, xs_test, ys_test, ws_test):
        loss = 100*(self.cls_loss + self.reg_loss) + self.obj_cls_acc
        self.train = self.optimizer.minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            cls_loss_history = []
            reg_loss_history = []
            obj_loss_history = []
            obj_acc_history = []
            for iter in range(self.max_iter):
                imgs_batch, msks_batch, labs_batch, xs_batch, ys_batch, ws_batch = sess.run(
                    [imgs_train, masks_train, labs_train, xs_train, ys_train, ws_train])
                _, _, cls_loss, reg_loss, obj_loss = sess.run([self.train, self.wd_op, self.cls_loss, self.reg_loss, self.obj_cls_loss],
                                                    {self.imgs: imgs_batch, self.msks: msks_batch, self.labs: labs_batch,
                                                     self.is_train: True, self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})
                print('Iter ', iter + 1, '\tRegion Cls Loss : ', cls_loss, '\t Region Reg Loss : ', reg_loss, '\t Object Cls Loss: ', obj_loss)
                cls_loss_history.append(cls_loss)
                reg_loss_history.append(reg_loss)
                obj_loss_history.append(obj_loss)

                if (iter+1)%self.epoch_size==0: # run a full test
                    obj_acc = 0
                    reg_loss = 0
                    cls_loss = 0
                    for i in range(self.epoch_size):
                        imgs_batch, msks_batch, labs_batch, xs_batch, ys_batch, ws_batch = sess.run(
                            [imgs_test, masks_test, labs_test, xs_test, ys_test, ws_test])
                        temp_obj_acc, temp_reg_loss, temp_cls_loss = sess.run([self.obj_cls_acc, self.reg_loss, self.cls_loss],
                                                     {self.imgs: imgs_batch, self.msks: msks_batch, self.labs: labs_batch,
                                                     self.is_train: False, self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})
                        obj_acc += temp_obj_acc
                        reg_loss += temp_reg_loss
                        cls_loss += temp_cls_loss

                        # if (i+1)%self.epoch_size==0:
                        #     imgs, imgs_cropped, proposals, masks = sess.run([self.imgs, self.imgs_cropped, self.cls_output, self.msks],
                        #                                   {self.imgs: imgs_batch, self.msks: msks_batch, self.is_train: False,
                        #                                    self.xs: xs_batch, self.ys: ys_batch, self.ws: ws_batch})
                        #     img = np.clip(np.squeeze(imgs[0,:,:,:])/256., 0, 1)
                        #     img_cropped = np.clip(np.squeeze(imgs_cropped[0,:,:,:])/256., 0, 1)
                        #     proposal = np.clip(np.squeeze(proposals[0,:,:,:])/2., 0, 1)
                        #     mask = np.clip(np.squeeze(masks[0,:,:,:])/2., 0, 1)
                        #
                        #     plt.figure(4)
                        #     plt.subplot(211)
                        #     plt.imshow(img)
                        #     plt.subplot(212)
                        #     plt.imshow(img_cropped)
                        #     plt.show()
                        #     plt.figure(5)
                        #     plt.subplot(211)
                        #     plt.imshow(proposal)
                        #     plt.subplot(212)
                        #     plt.imshow(mask)
                        #     plt.show()

                    obj_acc /= self.epoch_size
                    reg_loss /= self.epoch_size
                    cls_loss /= self.epoch_size

                    obj_acc_history.append(obj_acc)
                    print('Test Region Cls Loss: ', cls_loss, '\tRegion Reg Loss: ', reg_loss, '\tObj Cls Acc: ', obj_acc)

            coord.request_stop()
            coord.join(threads)

        # plot
        fig3 = plt.figure()

        # ax.plot(num_iter, loss_iter)
        plot_2D(cls_loss_history, fig3, '141',
                {'xlabel': '#Iteration', 'ylabel': 'Region Cls Loss', 'title': 'Faster-RCNN'})
        plot_2D(reg_loss_history, fig3, '142',
                {'xlabel': '#Iteration', 'ylabel': 'Region Reg Loss', 'title': 'Faster-RCNN'})
        plot_2D(obj_loss_history, fig3, '143',
                {'xlabel': '#Iteration', 'ylabel': 'Object Cls Loss', 'title': 'Faster-RCNN'})
        plot_2D(obj_acc_history, fig3, '144',
                {'xlabel': '#Iteration', 'ylabel': 'Object Cls Test Accuracy', 'title': 'Faster-RCNN'})
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
    # task 1: train proposal classifier
    # model.train_cls(imgs_train, masks_train, imgs_test, masks_test)

    # task 2: train region proposal networks (cls+reg)
    # model.train_rpn(imgs_train, masks_train, xs_train, ys_train, ws_train, imgs_test, masks_test, xs_test, ys_test, ws_test)

    # task 3: object detection pipeline TODO: RPN works, end2end classification problem
    model.train(imgs_train, labs_train, masks_train, xs_train, ys_train, ws_train, imgs_test, labs_test, masks_test, xs_test, ys_test, ws_test)


if __name__ =='__main__':
    main()
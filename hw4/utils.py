import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        img = tf.cast(img, tf.float32)

        img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1, batch_size=50, capacity=10000, allow_smaller_final_batch=True)
        '''batch can not be passed to graph directly, either sess.run to get value, or use in graph'''

    return img_batch, lab_batch

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np


def CRS(num=200000):
    f = h5py.File("./data/rectcrs_z.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')
    data = data*2-255. #(-255,255)

    label_key = f.keys()[1]
    label = np.asarray(f[label_key])
    z_key = f.keys()[2]
    z = np.asarray(f[z_key])
    z = z.astype('float32') * 2 - 1

    split = 0.1
    l = len(data)  # length of data
    n1 = int(split * l)  # split for testing
    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16] * 20  # sample(range(l), n1)

    x_test = data[indices]
    y_test = label[indices]
    z_test = z[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    z_train = np.delete(z, indices, 0)

    return (x_train, y_train), (x_test, y_test)


def CelebA_glass(num=200000):
    f = h5py.File("./data/celeba_64.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')
    data = data*2-255. #(-255,255)
    label = np.load('./data/glass_label.npy')

    indices = [0,1,2,3,4,5, 52,   92,  118,  143,  151,  153,  187,  188,  192,  201,  228,
        233,  236,  263,  265,  274,  275,  309,  329,  334,  349,  372,
        374,  382,  391,  443,  446,  450,  481,  483,  499,  510,  519,
        574,  580,  607,  618,  623,  627,  645,  672,  674,  675,  682,
        686,  704,  709,  719,  725,  759,  777,  787,  797,  814,  901,
        903,  906,  910,  925,  929,  934,  937,  943,  950,  957, 1027,
       1032, 1051, 1062, 1095, 1100, 1108, 1162, 1165, 1183, 1186, 1193,
       1214, 1223, 1240, 1302, 1324, 1350, 1351, 1399, 1400, 1434, 1440,
       1443, 1456, 1469, 1470, 1505, 1510, 1528, 1553, 1554, 1556, 1558,
       1597, 1605, 1607, 1612, 1615, 1651, 1674, 1689, 1702, 1720, 1727,
       1735, 1740, 1753, 1754] *10

    x_test = data[indices]
    y_test = label[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    return (x_train, y_train), (x_test, y_test)


def Mnist64(num=200000):
    aa = np.load('./data/Mnist4k_b.npy')  # range in (-1,1), 1 for digit pixels
    data = aa.item()['data']
    data = (data + 1) / 2
    label = aa.item()['label']
    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16] * 20
    x_test = data[indices]
    y_test = label[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    # range from 0 to 255
    return (np.expand_dims(x_train * 255, axis=1), y_train), (np.expand_dims(x_test * 255, axis=1), y_test)


def Mnist64_trans(num=200000):

    f = h5py.File("./data/Mnist64_100k_b_trans.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')
    data = data * 255
    label_key = f.keys()[1]
    label = np.asarray(f[label_key], dtype='float32')
    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16] * 20
    x_test = data[indices]
    y_test = label[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    return (x_train, y_train), (x_test, y_test)


def Mnist128_trans(num=200000):

    f = h5py.File("./dataMnist128_10k_b_trans.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')
    data = data * 255
    label_key = f.keys()[1]
    label = np.asarray(f[label_key], dtype='float32')
    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16] * 20
    x_test = data[indices]
    y_test = label[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    return (x_train, y_train), (x_test, y_test)

def Mnist64_switch(num=200000):

    f = h5py.File("./data/Mnist64_10k_switch.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')
    data = data * 255 #(-255,255)
    label_key = f.keys()[1]
    label = np.asarray(f[label_key], dtype='float32')

    # mask = []
    # for i, y_i in enumerate(label):
    #     if not y_i.any():
    #         mask += [False]
    #     else:
    #         mask += [True]
    # label = label[mask]
    # data = data[mask]

    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16] * 10
    x_test = data[indices]
    y_test = label[indices]
    x_train = np.delete(data, indices, 0)
    y_train = np.delete(label, indices, 0)
    return (x_train, y_train), (x_test, y_test)




import os
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png
        
        if len(paths) != 0:
            break

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)

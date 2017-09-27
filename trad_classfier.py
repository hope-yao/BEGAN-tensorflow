# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def lenet64(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):

    end_points = {}

    # with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
    net = slim.conv2d(images, 16, 5, scope='conv1')
    net = slim.max_pool2d(net, 2, 2, scope='pool1')
    net = slim.conv2d(net, 32, 5, scope='conv2')
    net = slim.max_pool2d(net, 2, 2, scope='pool2')
    net = slim.conv2d(net, 64, 5, scope='conv3')
    net = slim.max_pool2d(net, 2, 2, scope='pool3')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    mid_output = net = slim.fully_connected(net, 1024, scope='fc3')
    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
    #                    scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits)

    return mid_output, end_points




if __name__ == '__main__':
    from tqdm import tqdm
    import numpy as np
    from utils import *

    batch_size = 128
    n_class = 2
    ch_in = 1
    img_size = 64
    lr = 0.001

    x = tf.placeholder(tf.float32,shape=(batch_size, img_size, img_size, ch_in))
    y = tf.placeholder(tf.float32,shape=(batch_size, n_class))
    _, end_points = lenet64(x,num_classes=2,prediction_fn=tf.sigmoid)


    ## LOSS FUNCTION ##
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=end_points['Logits']))
    err = tf.reduce_mean(tf.abs(y-end_points['Predictions']))

    ## OPTIMIZER ##
    learning_rate = tf.Variable(lr) # learning rate for optimizer
    optimizer=tf.train.AdadeltaOptimizer(learning_rate)
    grads=optimizer.compute_gradients(err)
    # for i,(g,v) in enumerate(grads):
    #     if g is not None:
    #         grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
    train_op=optimizer.apply_gradients(grads)


    ## training starts ###
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    init = tf.global_variables_initializer()
    sess.run(init)

    from load_data import *
    (x_train, y_train), (x_test, y_test) = CRS()
    x_train = np.transpose(x_train,(0,2,3,1))
    x_test = np.transpose(x_test,(0,2,3,1))
    count = 0
    it_per_ep = int( len(x_train) / batch_size )
    for itr in range(5000):
        for i in tqdm(range(it_per_ep)):
            x_input = x_train[i*batch_size : (i + 1)*batch_size]
            y_input = y_train[i*batch_size : (i + 1)*batch_size]
            feed_dict_train = {x: x_input, y: y_input}
            results = sess.run(train_op, feed_dict_train)

            if count % 100 == 0:
                rand_idx = np.random.random_integers(0,len(x_test)-1,size=batch_size)
                x_input = x_test[rand_idx]
                y_input = y_test[rand_idx]
                feed_dict_test = {x: x_input, y: y_input}
                train_result = sess.run(err, feed_dict_train)
                test_result = sess.run(err, feed_dict_test)
                print("iter={} : train_err_last: {}, test_err_last: {}" .format(count, train_result, test_result))

                if 0:
                    nn = np.load('/home/exx/Documents/Hope/rec_crs.npy')
                    crs = nn.item()['cross_img']
                    crs = np.reshape(crs, (crs.shape[0], 64, 64))
                    x_input = crs[i * batch_size: (i + 1) * batch_size]
                    feed_dict_train = {x: np.expand_dims(x_input, 3), y: [[1, 0]] * batch_size}
                    results = sess.run(end_points['Predictions'], feed_dict_train)
            # if itr % 1000 == 0:
            #     sess.run(tf.assign(learning_rate, learning_rate * 0.5))
            count += 1
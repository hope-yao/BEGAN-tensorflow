import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from utils import *

def GeneratorCNN(y_data, z, z_num_sep, hidden_num, output_num, repeat_num ):
    bs, n_subnet = y_data.get_shape().as_list()
    assert (n_subnet==len(z_num_sep))
    imsize =28
    out = tf.zeros((bs,output_num,imsize,imsize))
    out_sub = []
    for net_i,_ in enumerate(z_num_sep):
        z_i = tf.slice(z, (0, sum(z_num_sep[:net_i])), (bs, z_num_sep[net_i]))
        x_i = slim.fully_connected(z_i, np.prod([7, 7, hidden_num[net_i]]), activation_fn=None)
        x_i = reshape(x_i, 7, 7, hidden_num[net_i])
        for idx in range(repeat_num):
            x_i = slim.conv2d_transpose(x_i, (idx+1)*hidden_num[net_i], 3, 2, activation_fn=tf.nn.elu, data_format='NCHW')
            # x_i = slim.conv2d(x_i, hidden_num[net_i], 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num[net_i], 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # if idx < repeat_num - 1:
            #     x_i = upscale(x_i, 2, data_format)
        out_i = slim.conv2d(x_i, output_num, 3, 1, activation_fn=None, data_format='NCHW')
        y_i = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(y_data[:, net_i], 1), 1), 1), [1, 1, imsize, imsize])
        out += y_i * out_i
        out_sub += [out_i]
    # out = tf.clip_by_value(out, 0, 1)
    return out, out_sub

def Encoder(x, z_num, repeat_num, hidden_num):

    # Encoder
    for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format='NCHW')
        # x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        # if idx < repeat_num - 1:
            # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)
    x = tf.reshape(x, [-1, np.prod([7, 7, channel_num])])
    z = slim.fully_connected(x, z_num, activation_fn=None) # times 2 for mean and variance

    return  z

def Decoder(y_data, z, input_channel, z_num_sep, repeat_num, hidden_num):
    imsize = 28
    bs, n_subnet = y_data.get_shape().as_list()
    dup = 2 #2 if without x_mid, which is only x_real and x_fake. Doubled to 4 if with x_mid
    # Decoder
    out = tf.zeros((bs*dup, input_channel, imsize, imsize))
    out_sub = []
    for net_i in range(n_subnet):
        z_i = tf.slice(z, (0, sum(z_num_sep[:net_i])), (bs*dup, z_num_sep[net_i]))
        x_i = slim.fully_connected(z_i, np.prod([7, 7, hidden_num[net_i]]), activation_fn=None)
        x_i = reshape(x_i, 7, 7, hidden_num[net_i])
        for idx in range(repeat_num):
            x_i = slim.conv2d_transpose(x_i, (idx+1)*hidden_num[net_i], 3, 2, activation_fn=tf.nn.elu, data_format='NCHW')
            # x_i = slim.conv2d(x_i, hidden_num[net_i], 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # x_i = slim.conv2d(x_i, hidden_num[net_i], 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            # if idx < repeat_num - 1:
            #     x_i = upscale(x_i, 2, data_format)
        out_i = slim.conv2d(x_i, input_channel, 3, 1, activation_fn=None, data_format='NCHW')
        y_i = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(y_data[:, net_i], 1), 1), 1), [dup, 1, imsize, imsize])
        out += y_i * out_i
        out_sub += [out_i[0:bs*2]] #only output the first random sample

    return out, out_sub


def calc_eclipse_loss_analy(dz,z,z_num_sep):
    dzf, dzr = tf.split(dz,2)
    l_f = []
    l_r = []
    m_r = []
    m_f = []
    dim1, dim2 = dzf.get_shape().as_list()
    for neti,z_num_i in enumerate(z_num_sep):
        dzf_i = tf.slice(dzf,(0,sum(z_num_sep[:neti])),(dim1,z_num_sep[neti]))
        dzr_i = tf.slice(dzr,(0,sum(z_num_sep[:neti])),(dim1,z_num_sep[neti]))
        z_i = tf.slice(z,(0,sum(z_num_sep[:neti])),(dim1,z_num_sep[neti]))

        dzf_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzf_i, 0), 0), (dim1, 1))
        dzf_i = dzf_i - (dzf_mean+1e-8)
        dzr_mean = tf.tile(tf.expand_dims(tf.reduce_mean(dzr_i, 0), 0), (dim1, 1))
        dzr_i = dzr_i - (dzr_mean+1e-8)
        z_mean = tf.tile(tf.expand_dims(tf.reduce_mean(z_i, 0), 0), (dim1, 1))
        z_i = z_i - (z_mean+1e-8)

        mi_dzf = tf.matmul(tf.transpose(dzf_i, (1, 0)), dzf_i) / dim1
        mi_dzr = tf.matmul(tf.transpose(dzr_i, (1, 0)), dzr_i) / dim1
        mi_z = tf.eye(z_num_i,z_num_i)

        # res_det_f_i = tf.square(tf.matrix_determinant(mi_z) - tf.matrix_determinant(mi_dzf))
        # res_det_r_i = tf.square(tf.matrix_determinant(mi_z) - tf.matrix_determinant(mi_dzr))

        res_f_i = tf.square(mi_z-mi_dzf)
        l_f_i = tf.reduce_mean(res_f_i) + tf.reduce_mean(tf.diag_part(res_f_i))#+res_det_f_i ##
        l_f += [tf.expand_dims(l_f_i, 0)]
        res_r_i = tf.square(mi_z - mi_dzr)
        l_r_i = tf.reduce_mean(res_r_i) + tf.reduce_mean(tf.diag_part(res_r_i))#+res_det_r_i ##
        l_r += [tf.expand_dims(l_r_i, 0)]

        m_f_i = tf.reduce_mean(tf.square(z_mean - dzf_mean))
        m_f += [tf.expand_dims(m_f_i, 0)]
        m_r_i = tf.reduce_mean(tf.square(z_mean - dzr_mean))
        m_r += [tf.expand_dims(m_r_i, 0)]

        if neti==0:
            tmp = mi_dzf

    return  l_f, l_r, m_r, m_f, tf.reduce_mean(tmp)#, (mi_z, mi_dzf, dzf_i)

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c):
    # if data_format == 'NCHW':
    x = tf.reshape(x, [-1, c, h, w])
    # else:
    # x = tf.reshape(x, [-1, h, w, c])
    return x


def calc_pt_Angular(z_d_gen, bs):
    nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
    denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
    pt = tf.abs(tf.transpose((nom / denom), (1, 0)) / denom)
    pt = pt - tf.diag(tf.diag_part(pt))
    pulling_term = tf.reduce_sum(pt) / (bs * (bs - 1))
    return pulling_term, nom, denom
    #
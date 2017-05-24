import numpy as np
import tensorflow as tf
slim = tf.contrib.slim


def BEGAN_enc(input, hidden_num=128, z_num=64, repeat_num=4, data_format='NCHW', reuse=False):
    x = slim.conv2d(input, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

    prev_channel_num = hidden_num
    for idx in range(repeat_num):
        channel_num = hidden_num * (idx + 1)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        if idx < repeat_num - 1:
            # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
            x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)

    x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
    z = x = slim.fully_connected(x, z_num, activation_fn=None)
    return z

def BEGAN_dec(input, hidden_num, input_channel=3, data_format='NCHW', repeat_num=4):
    x = slim.fully_connected(input, np.prod([8, 8, hidden_num]), activation_fn=None)
    x = reshape(x, 8, 8, hidden_num, data_format)

    for idx in range(repeat_num):
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        if idx < repeat_num - 1:
            x = upscale(x, 2, data_format)

    out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
    return out

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        x = slim.fully_connected(z, np.prod([8, 8, hidden_num]), activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, output_num, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # interpolation
        start_z = tf.slice(z, [16+4, 0], [1, 64])
        end_z = tf.slice(z, [16+3, 0], [1, 64])
        intp_z1 = start_z
        num = 16
        for ci in range(1, num, 1):
            intp_z1 = tf.concat([intp_z1, start_z + (end_z - start_z) * ci/(num-1)], 0)
        inc_z0 = -(end_z - start_z) * 0
        num = 16
        for ci in range(1, num, 1):
            inc_z0 = tf.concat([inc_z0, -(end_z - start_z) * ci / (num - 1)], 0)
        inc_z1 = (end_z - start_z) * 0
        num = 16
        for ci in range(1, num, 1):
            inc_z1 = tf.concat([inc_z1, (end_z - start_z) * ci / (num - 1)], 0)
        morph0 = tf.slice(z, [16, 0], [16, 64])-(end_z - start_z)
        morph1 = tf.slice(z, [16, 0], [16, 64])+(end_z - start_z)

        start_z = tf.slice(z, [16+4, 0], [1, 64])
        end_z = tf.slice(z, [16+5, 0], [1, 64])
        intp_z2 = start_z
        num = 16
        for ci in range(1, num, 1):
            intp_z2 = tf.concat([intp_z2, start_z + (end_z - start_z) * ci/(num-1)], 0)
        inc_z2 = -(end_z - start_z) * 0
        num = 16
        for ci in range(1, num, 1):
            inc_z2 = tf.concat([inc_z2, -(end_z - start_z) * ci / (num - 1)], 0)
        morph2 = tf.slice(z, [16, 0], [16, 64])+(end_z - start_z)

        # extrapolation
        start_z = tf.slice(z, [16+4, 0], [1, 64])
        end_z = tf.slice(z, [16+3, 0], [1, 64])
        extp_z1 = start_z
        num = 16
        for ci in range(1, num, 1):
            extp_z1 = tf.concat([extp_z1, start_z - (end_z - start_z) * ci/(num-1)], 0)
        extp_z11 = end_z
        num = 16
        for ci in range(1, num, 1):
            extp_z11 = tf.concat([extp_z11, end_z - (start_z-end_z) * ci/(num-1)], 0)

        start_z = tf.slice(z, [16+4, 0], [1, 64])
        end_z = tf.slice(z, [16+5, 0], [1, 64])
        extp_z2 = start_z
        num = 16
        for ci in range(1, num, 1):
            extp_z2 = tf.concat([extp_z2, start_z - (end_z - start_z) * ci/(num-1)], 0)
        extp_z21 = end_z
        num = 16
        for ci in range(1, num, 1):
            extp_z21 = tf.concat([extp_z21, end_z - (start_z-end_z) * ci/(num-1)], 0)

        x = tf.concat([x,intp_z1,extp_z1,extp_z11,intp_z2,extp_z2,extp_z21,inc_z0,inc_z1,inc_z2,morph0,morph1,morph2],0)

        # Decoder
        x = slim.fully_connected(x, np.prod([8, 8, hidden_num]), activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

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

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)

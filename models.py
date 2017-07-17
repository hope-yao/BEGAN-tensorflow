import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

# def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
#     with tf.variable_scope("G", reuse=reuse) as vs:
#         x = slim.fully_connected(z, np.prod([13, 13, hidden_num]), activation_fn=None)
#         x = reshape(x, 13, 13, hidden_num, data_format)
#
#         for idx in range(repeat_num):
#             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             if idx < repeat_num - 1:
#                 x = upscale(x, 2, data_format)
#
#         out = slim.conv2d(x, output_num, 3, 1, activation_fn=None, data_format=data_format)
#
#     variables = tf.contrib.framework.get_variables(vs)
#     return out, variables

# def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
#     with tf.variable_scope("D") as vs:
#         # Encoder
#         x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#
#         prev_channel_num = hidden_num
#         for idx in range(repeat_num):
#             channel_num = hidden_num * (idx + 1)
#             x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             if idx < repeat_num - 1:
#                 # x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
#                 x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID', data_format=data_format)
#
#         x = tf.reshape(x, [-1, np.prod([13, 13, channel_num])])
#         z = x = slim.fully_connected(x, z_num, activation_fn=None)
#
#         # Decoder
#         x = slim.fully_connected(x, np.prod([13, 13, hidden_num]), activation_fn=None)
#         x = reshape(x, 13, 13, hidden_num, data_format)
#
#         for idx in range(repeat_num):
#             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
#             if idx < repeat_num - 1:
#                 x = upscale(x, 2, data_format)
#
#         out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
#
#     variables = tf.contrib.framework.get_variables(vs)
#     return out, z, variables

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


def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
    with tf.variable_scope("G", reuse=reuse) as vs:
        encoded = Dense(8 * 128)(z)
        encoded = Dense(13 * 13 * 128)(encoded)
        encoded = Reshape((13, 13, 128))(encoded)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    variables = tf.contrib.framework.get_variables(vs)
    return tf.transpose(decoded,(0,3,1,2)), variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        input_img = tf.transpose(x,(0,2,3,1))

        from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
        # input_img = Input(shape=(104, 104, 1))  # adapt this if using `channels_first` image data format

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        encoded = Flatten()(encoded)
        encoded = Dense(8 * 128)(encoded)
        z = encoded = Dense(z_num)(encoded)
        encoded = Dense(8 * 128)(encoded)
        encoded = Dense(13 * 13 * 128)(encoded)
        encoded = Reshape((13, 13, 128))(encoded)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)  #z1
        z1 = x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)     #z2
        z2 = x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        z3 = x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)

        variables = tf.contrib.framework.get_variables(vs)
        return tf.transpose(decoded,(0,3,1,2)), z, z1, z2, z3, variables




def gram_enc(x, gram_dim=128):
    with tf.variable_scope("gram_embedding") as vs:
        input_img = tf.transpose(x,(0,2,3,1))

        from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        encoded = Flatten()(encoded)
        encoded = Dense(2048)(encoded)
        z = encoded = Dense(gram_dim)(encoded)
        variables = tf.contrib.framework.get_variables(vs)

        return z, variables

def gram_dec(z):
    with tf.variable_scope("gram_gen") as vs:
        from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

        encoded = Dense(2048)(z)
        encoded = Dense(16 * 16* 32)(encoded)
        encoded = Reshape((16, 16, 32))(encoded)

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  #z1
        z1 = x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(24, (3, 3), activation='relu', padding='same')(x)     #z2
        z2 = x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        z3 = x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        z4 = x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation=None, padding='same')(x)

        variables = tf.contrib.framework.get_variables(vs)
        return tf.transpose(decoded,(0,3,1,2)), z, z1, z2, z3, z4, variables


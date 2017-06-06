from keras import backend
import tensorflow as tf
import h5py


def conv2d(x, W, stride, padding="SAME"):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)


def max_pool(x, k_size, stride, padding="SAME"):
    # use avg pooling instead, as described in the paper
    return tf.nn.avg_pool(x, ksize=[1, k_size, k_size, 1],
                          strides=[1, stride, stride, 1], padding=padding)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    #     features_mean,features_var =tf.nn.moments(features,axes=[0])
    features_mean = tf.reduce_mean(features, 0)
    features = (features - features_mean) / 1
    gram = backend.dot(features, backend.transpose(features))
    return gram


def style_loss(style, combination, weight):
    mb_size = style.shape[0].value
    width = style.shape[1].value
    height = style.shape[2].value

    loss_temp = 0.
    channels = 3
    size = height * width

    # C = S =[]
    # for i in range(mb_size):
    #     C += [gram_matrix(combination[i])]
    # for j in range(mb_size):
    #     S += [gram_matrix(style[j])]
    # for i in range(mb_size):
    #     for j in range(mb_size):
    #         loss_temp = tf.add(loss_temp, backend.sum(backend.square(S[j] - C[i])) / (4. * (channels ** 2) * (size ** 2) * weight[i,j] ))
    # return loss_temp*1e-6
    for i in range(mb_size):
        C = gram_matrix(combination[i])
        S = gram_matrix(style[i])
        loss_temp = tf.add(loss_temp, backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))) * 1e-7
    return loss_temp

def load_vgg(vgg_filepath):
    f = h5py.File(vgg_filepath,'r')
    ks = f.keys()

    vgg16_weights=[]
    vgg16_bias=[]
    for i in range(18):
        if (len(f[ks[i]].values())) != 0:
            vgg16_weights.append(f[ks[i]].values()[0][:])
            vgg16_bias.append(f[ks[i]].values()[1][:])
        else:
            continue

    W_conv1 = (tf.constant(vgg16_weights[0]))
    W_conv2 = (tf.constant(vgg16_weights[1]))
    W_conv3 = (tf.constant(vgg16_weights[2]))
    W_conv4 = (tf.constant(vgg16_weights[3]))
    W_conv5 = (tf.constant(vgg16_weights[4]))
    W_conv6 = (tf.constant(vgg16_weights[5]))
    W_conv7 = (tf.constant(vgg16_weights[6]))
    W_conv8 = (tf.constant(vgg16_weights[7]))
    W_conv9 = (tf.constant(vgg16_weights[8]))
    W_conv10= (tf.constant(vgg16_weights[9]))
    W_conv11= (tf.constant(vgg16_weights[10]))
    W_conv12= (tf.constant(vgg16_weights[11]))
    W_conv13= (tf.constant(vgg16_weights[12]))

    b_conv1 = tf.reshape(tf.constant(vgg16_bias[0]),[-1])
    b_conv2 = tf.reshape(tf.constant(vgg16_bias[1]),[-1])
    b_conv3 = tf.reshape(tf.constant(vgg16_bias[2]),[-1])
    b_conv4 = tf.reshape(tf.constant(vgg16_bias[3]),[-1])
    b_conv5 = tf.reshape(tf.constant(vgg16_bias[4]),[-1])
    b_conv6 = tf.reshape(tf.constant(vgg16_bias[5]),[-1])
    b_conv7 = tf.reshape(tf.constant(vgg16_bias[6]),[-1])
    b_conv8 = tf.reshape(tf.constant(vgg16_bias[7]),[-1])
    b_conv9 = tf.reshape(tf.constant(vgg16_bias[8]),[-1])
    b_conv10 = tf.reshape(tf.constant(vgg16_bias[9]),[-1])
    b_conv11 = tf.reshape(tf.constant(vgg16_bias[10]),[-1])
    b_conv12 = tf.reshape(tf.constant(vgg16_bias[11]),[-1])
    b_conv13 = tf.reshape(tf.constant(vgg16_bias[12]),[-1])

    W = [W_conv1,W_conv2,W_conv3,W_conv4,W_conv5,W_conv6,W_conv7,W_conv8,W_conv9,W_conv10,W_conv11,W_conv12,W_conv13]
    b = [b_conv1,b_conv2,b_conv3,b_conv4,b_conv5,b_conv6,b_conv7,b_conv8,b_conv9,b_conv10,b_conv11,b_conv12,b_conv13]
    return W, b

def gen_activation(combination_image, W_vgg, b_vgg):
    W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, W_conv9, W_conv10, W_conv11, W_conv12, W_conv13 = W_vgg
    b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_conv6, b_conv7, b_conv8, b_conv9, b_conv10, b_conv11, b_conv12, b_conv13 = b_vgg
    ######### block 1 ########
    conv_out1 = conv2d(combination_image, W_conv1, stride=1, padding='SAME')
    conv_out1 = tf.nn.bias_add(conv_out1, b_conv1)
    conv_out1 = tf.nn.relu(conv_out1)

    conv_out2 = conv2d(conv_out1, W_conv2, stride=1, padding='SAME')
    conv_out2 = tf.nn.bias_add(conv_out2, b_conv2)
    conv_out2 = tf.nn.relu(conv_out2)
    conv_out2 = max_pool(conv_out2, k_size=2, stride=2, padding="SAME")

    ######### block 2 ########
    conv_out3 = conv2d(conv_out2, W_conv3, stride=1, padding='SAME')
    conv_out3 = tf.nn.bias_add(conv_out3, b_conv3)
    conv_out3 = tf.nn.relu(conv_out3)

    conv_out4 = conv2d(conv_out3, W_conv4, stride=1, padding='SAME')
    conv_out4 = tf.nn.bias_add(conv_out4, b_conv4)
    conv_out4 = tf.nn.relu(conv_out4)
    conv_out4 = max_pool(conv_out4, k_size=2, stride=2, padding="SAME")

    ######### block 3 ########
    conv_out5 = conv2d(conv_out4, W_conv5, stride=1, padding='SAME')
    conv_out5 = tf.nn.bias_add(conv_out5, b_conv5)
    conv_out5 = tf.nn.relu(conv_out5)

    conv_out6 = conv2d(conv_out5, W_conv6, stride=1, padding='SAME')
    conv_out6 = tf.nn.bias_add(conv_out6, b_conv6)
    conv_out6 = tf.nn.relu(conv_out6)

    conv_out7 = conv2d(conv_out6, W_conv7, stride=1, padding='SAME')
    conv_out7 = tf.nn.bias_add(conv_out7, b_conv7)
    conv_out7 = tf.nn.relu(conv_out7)
    conv_out7 = max_pool(conv_out7, k_size=2, stride=2, padding="SAME")

    ######### block 4 ########
    conv_out8 = conv2d(conv_out7, W_conv8, stride=1, padding='SAME')
    conv_out8 = tf.nn.bias_add(conv_out8, b_conv8)
    conv_out8 = tf.nn.relu(conv_out8)

    conv_out9 = conv2d(conv_out8, W_conv9, stride=1, padding='SAME')
    conv_out9 = tf.nn.bias_add(conv_out9, b_conv9)
    conv_out9 = tf.nn.relu(conv_out9)

    conv_out10 = conv2d(conv_out9, W_conv10, stride=1, padding='SAME')
    conv_out10 = tf.nn.bias_add(conv_out10, b_conv10)
    conv_out10 = tf.nn.relu(conv_out10)
    conv_out10 = max_pool(conv_out10, k_size=2, stride=2, padding="SAME")

    ######### block 5 ########
    conv_out11 = conv2d(conv_out10, W_conv11, stride=1, padding='SAME')
    conv_out11 = tf.nn.bias_add(conv_out11, b_conv11)
    conv_out11 = tf.nn.relu(conv_out11)

    conv_out12 = conv2d(conv_out11, W_conv12, stride=1, padding='SAME')
    conv_out12 = tf.nn.bias_add(conv_out12, b_conv12)
    conv_out12 = tf.nn.relu(conv_out12)

    conv_out13 = conv2d(conv_out12, W_conv13, stride=1, padding='SAME')
    conv_out13 = tf.nn.bias_add(conv_out13, b_conv12)
    conv_out13 = tf.nn.relu(conv_out13)

    gen_conv_out = [conv_out1,conv_out2,conv_out3,conv_out4,conv_out5,conv_out6,conv_out7,conv_out8,conv_out9,conv_out10,conv_out11,conv_out12,conv_out13]
    return gen_conv_out

def true_activation(style_image, W_vgg, b_vgg):
    W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_conv6, W_conv7, W_conv8, W_conv9, W_conv10, W_conv11, W_conv12, W_conv13 = W_vgg
    b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_conv6, b_conv7, b_conv8, b_conv9, b_conv10, b_conv11, b_conv12, b_conv13 = b_vgg
    ######### block 1 ########
    conv_out1_S = conv2d(style_image, W_conv1, stride=1, padding='SAME')
    conv_out1_S = tf.nn.bias_add(conv_out1_S, b_conv1)
    conv_out1_S = tf.nn.relu(conv_out1_S)

    conv_out2_S = conv2d(conv_out1_S, W_conv2, stride=1, padding='SAME')
    conv_out2_S = tf.nn.bias_add(conv_out2_S, b_conv2)
    conv_out2_S = tf.nn.relu(conv_out2_S)
    conv_out2_S = max_pool(conv_out2_S, k_size=2, stride=2, padding="SAME")

    ######### block 2 ########
    conv_out3_S = conv2d(conv_out2_S, W_conv3, stride=1, padding='SAME')
    conv_out3_S = tf.nn.bias_add(conv_out3_S, b_conv3)
    conv_out3_S = tf.nn.relu(conv_out3_S)

    conv_out4_S = conv2d(conv_out3_S, W_conv4, stride=1, padding='SAME')
    conv_out4_S = tf.nn.bias_add(conv_out4_S, b_conv4)
    conv_out4_S = tf.nn.relu(conv_out4_S)
    conv_out4_S = max_pool(conv_out4_S, k_size=2, stride=2, padding="SAME")

    ######### block 3 ########
    conv_out5_S = conv2d(conv_out4_S, W_conv5, stride=1, padding='SAME')
    conv_out5_S = tf.nn.bias_add(conv_out5_S, b_conv5)
    conv_out5_S = tf.nn.relu(conv_out5_S)

    conv_out6_S = conv2d(conv_out5_S, W_conv6, stride=1, padding='SAME')
    conv_out6_S = tf.nn.bias_add(conv_out6_S, b_conv6)
    conv_out6_S = tf.nn.relu(conv_out6_S)

    conv_out7_S = conv2d(conv_out6_S, W_conv7, stride=1, padding='SAME')
    conv_out7_S = tf.nn.bias_add(conv_out7_S, b_conv7)
    conv_out7_S = tf.nn.relu(conv_out7_S)
    conv_out7_S = max_pool(conv_out7_S, k_size=2, stride=2, padding="SAME")

    ######### block 4 ########
    conv_out8_S = conv2d(conv_out7_S, W_conv8, stride=1, padding='SAME')
    conv_out8_S = tf.nn.bias_add(conv_out8_S, b_conv8)
    conv_out8_S = tf.nn.relu(conv_out8_S)

    conv_out9_S = conv2d(conv_out8_S, W_conv9, stride=1, padding='SAME')
    conv_out9_S = tf.nn.bias_add(conv_out9_S, b_conv9)
    conv_out9_S = tf.nn.relu(conv_out9_S)

    conv_out10_S = conv2d(conv_out9_S, W_conv10, stride=1, padding='SAME')
    conv_out10_S = tf.nn.bias_add(conv_out10_S, b_conv10)
    conv_out10_S = tf.nn.relu(conv_out10_S)
    conv_out10_S = max_pool(conv_out10_S, k_size=2, stride=2, padding="SAME")

    ######### block 5 ########
    conv_out11_S = conv2d(conv_out10_S, W_conv11, stride=1, padding='SAME')
    conv_out11_S = tf.nn.bias_add(conv_out11_S, b_conv11)
    conv_out11_S = tf.nn.relu(conv_out11_S)

    conv_out12_S = conv2d(conv_out11_S, W_conv12, stride=1, padding='SAME')
    conv_out12_S = tf.nn.bias_add(conv_out12_S, b_conv12)
    conv_out12_S = tf.nn.relu(conv_out12_S)

    conv_out13_S = conv2d(conv_out12_S, W_conv13, stride=1, padding='SAME')
    conv_out13_S = tf.nn.bias_add(conv_out13_S, b_conv13)
    conv_out13_S = tf.nn.relu(conv_out13_S)

    style_conv_out = [conv_out1_S, conv_out2_S, conv_out3_S, conv_out4_S, conv_out5_S, conv_out6_S, conv_out7_S, conv_out8_S, conv_out9_S,
                    conv_out10_S, conv_out11_S, conv_out12_S, conv_out13_S]
    return style_conv_out


def total_style_cost(combination_image, style_image, z1, z2, bs):
    # Style transfer
    W_vgg, b_vgg = load_vgg('/home/doi5/Documents/Hope/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    gen_conv_out = gen_activation(combination_image, W_vgg, b_vgg)
    style_conv_out = true_activation(style_image, W_vgg, b_vgg)

    [conv_out1, conv_out2, conv_out3, conv_out4, conv_out5, conv_out6, conv_out7, conv_out8, conv_out9,
     conv_out10, conv_out11, conv_out12, conv_out13] = gen_conv_out
    [conv_out1_S, conv_out2_S, conv_out3_S, conv_out4_S, conv_out5_S, conv_out6_S, conv_out7_S, conv_out8_S, conv_out9_S,
                    conv_out10_S, conv_out11_S, conv_out12_S, conv_out13_S] = style_conv_out

    dd = tf.tile(tf.expand_dims(z1, 1), [1, bs, 1]) - tf.tile(tf.expand_dims(z2, 0), [bs, 1, 1])
    weight = tf.sqrt(tf.reduce_sum(tf.square(dd), 2))  # dist[i,j] = z1[i]-z2[j]
    # weight = weight / tf.tile(tf.expand_dims(tf.reduce_sum(dist, 1), 1),[1, 16])  # row-wise summation, duplicate to matrix, normalize

    sl1 = style_loss(conv_out2_S, conv_out2, weight)
    sl2 = style_loss(conv_out4_S, conv_out4, weight)
    sl3 = style_loss(conv_out7_S, conv_out7, weight)
    sl4 = style_loss(conv_out10_S, conv_out10, weight)
    sl = sl1 + sl2 + sl3 + sl4

    # return sl
    return 1/(1-tf.log(sl))

import numpy as np

ii = range(3, 32, 2)
ww = []
for i in ii:
    ww += [tf.Variable(np.ones((i, i, 1, 1),dtype='float32'))]
def white_style(input):
    result = tf.Variable(0.)
    count = 0
    for w in ww:
        x = input
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
        x = tf.clip_by_value(x, 0, 1)
        # result_ch = tf.sigmoid((tf.reduce_mean(x,axis=(1,2,3)) * (3+count*2))) -0.5
        # result +=tf.reduce_mean(tf.clip_by_value(-result_ch,-1,1))
        result_ch = tf.sigmoid((tf.reduce_mean(x, axis=(1, 2, 3)) * (3 + count * 2)))
        result += tf.reduce_sum(1-result_ch)
        count += 1
    return result



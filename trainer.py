from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def  material104():
    import scipy.io as sio
    WB = sio.loadmat('/home/hope-yao/Documents/Data/material/WB_sm.mat')['WB_sm']#[0:128]
    x_train = WB.astype('float32') * 255.
    x_train = np.reshape(x_train, (x_train.shape[0], 100, 100, 1))  # adapt this if using `channels_first` image data format
    X_train = np.ones((x_train.shape[0], 104, 104, 1))
    X_train[:, 2:102, 2:102, :] = x_train
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    y_train = np.eye(WB.shape[0])
    return (X_train, y_train), (X_train, y_train)


def CelebA(datadir, num=200000):
    '''load human face dataset'''
    import h5py
    from random import sample
    import numpy as np
    # f = h5py.File("rectcrs_z.hdf5", "r")
    f = h5py.File(datadir + "/Ti_hope.hdf5", "r")
    data_key = f.keys()[0]
    data = np.asarray(f[data_key], dtype='float32')  # normalized into (-1, 1)
    # data = (np.asarray(f[data_key],dtype='float32') / 255. - 0.5 )*2 # normalized into (-1, 1)
    data = data * 255
    data = data.transpose((0, 3, 2, 1))

    split = 0.1
    l = len(data)  # length of data
    n1 = int(split * l)  # split for testing
    indices = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16]  # sample(range(l), n1)

    x_test = data[indices]
    x_train = np.delete(data, indices, 0)

    return (x_train, 0), (x_test, 0)


import dateutil.tz
import datetime


def sample_simplex(y):
    '''sample inside the simple spanned by matrix y's row vectors'''
    from numpy.random import rand
    sample = []
    bs = y.shape[0]
    dim = bs - 1
    for i in range(bs):
        beta = rand(1,dim)
        beta = beta / np.sum(beta)   # evenly sample ON the simplex. all ranges from 0 to 1 and sum up to 1

        alpha = 0.6 # how away from the corner
        d = alpha * rand(1) # making sample INSIDE the simplex. this implementation is slightly concentrating to the edge
        # d = 1 # sample ON the simplex
        beta = beta * d

        sample_i = np.dot(np.insert(beta, i, 1.-alpha, axis=1),y) #used i here for some unknown reason....
        sample += [sample_i[0]]
    return np.asarray(sample)


def calc_pt(z_d_gen, batch_size):
    nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
    denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
    pt = tf.square(tf.transpose((nom / denom), (1, 0)) / denom)
    pt = pt - tf.diag(tf.diag_part(pt))
    pulling_term = tf.reduce_sum(pt) / (batch_size * (batch_size - 1))
    return pulling_term


def creat_dir(network_type):
    """code from on InfoGAN"""
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_log_dir = "logs/" + network_type
    exp_name = network_type + "_%s" % timestamp
    log_dir = os.path.join(root_log_dir, exp_name)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    root_model_dir = "models/" + network_type
    exp_name = network_type + "_%s" % timestamp
    model_dir = os.path.join(root_model_dir, exp_name)

    for path in [log_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    return log_dir, model_dir



class Trainer(object):
    def __init__(self, config):

        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.logdir, self.modeldir = creat_dir('GAN')
        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        height = width = 104
        self.channel = 1
        self.repeat_num = 4

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.modeldir)

        sv = tf.train.Supervisor(logdir=self.modeldir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        # self.sess = tf.Session()
        # self.saver.restore(self.sess, "./models/GAN/GAN_2017_06_23_14_09_45/model.ckpt-7649")

        # (self.X_train, self.y_train), (self.X_test, self.y_test) = material104()
        # x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        # y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        # feed_dict_fix = {self.x: x_input_fix,
        #                  self.z: np.asarray([[(15 - i) / 15., 0, 0, i / 15.] + [0] * 996 for i in range(16)]),
        #                  self.x_fix: x_input_fix[0:1]}
        #
        # x_img, x_rec, g_img, g_rec = \
        #     self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G], feed_dict_fix)
        # nrow = self.batch_size
        # all_G_z = np.concatenate([x_input_fix.transpose((0, 2, 3, 1)), x_rec, g_img, g_rec])
        # save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, 0), nrow=nrow)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        counter = 0
        from tqdm import tqdm
        (self.X_train, self.y_train), (self.X_test, self.y_test) = material104()
        x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        feed_dict_fix = {self.x: x_input_fix, self.z: y_input_fix, self.x_fix:x_input_fix[0:1]}

        for epoch in range(5000):
            it_per_ep = len(self.X_train) / self.batch_size
            for i in tqdm(range(it_per_ep)):
                counter += 1
                x_input = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_input = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.x: x_input, self.z: sample_simplex(y_input), self.x_fix:x_input_fix[0:1]}
                result = self.sess.run([self.d_loss,self.g_loss,self.measure,self.k_update,self.k_t,
                                        self.style_loss, self.pulling_term, self.pulling_term1, self.pulling_term2, self.pulling_term3, self.d_fvr],feed_dict)
                print(result)

                if counter in [3e4, 6e5, 12e5, 15e5]:
                    self.sess.run([self.g_lr_update, self.d_lr_update])

                if counter % 10 == 0:
                    x_img, x_rec, g_img, g_rec = \
                        self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G], feed_dict_fix)
                    nrow = self.batch_size
                    all_G_z = np.concatenate([x_input_fix.transpose((0,2,3,1)), x_rec, g_img, g_rec])
                    save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, counter),nrow=nrow)

                if counter in [1e2, 1e3, 5e3, 1e4, 2e4, 3e4, 1e5, 2e5]:
                    snapshot_name = "%s_%s" % ('experiment', str(counter))
                    fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                    print("Model saved in file: %s" % fn)

                if counter % 10 == 0:
                    summary = self.sess.run(self.summary_op, feed_dict)
                    self.summary_writer.add_summary(summary, counter)
                    self.summary_writer.flush()



    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, 1, 104, 104])
        self.normx = norm_img(self.x)
        self.x_img = denorm_img(self.normx,self.data_format)
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_num])

        self.x_fix = tf.placeholder(tf.float32, [1, 1, 104, 104])
        fix_img = tf.tile(self.x_fix[0:1], (self.batch_size, 1, 1, 1))

        # self.z = tf.random_uniform(
        #         (tf.shape(self.normx)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, self.G_var = GeneratorCNN(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)

        d_out, self.D_z, self.D_z1, self.D_z2, self.D_z3, self.D_var = DiscriminatorCNN(
                tf.concat([G, self.normx, fix_img], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format)
        AE_G, AE_x, _ = tf.split(d_out, 3)

        from keras.layers import Flatten
        z_d_gen, z_d_real, z_d_gen0  = tf.split(self.D_z, 3)
        self.pulling_term =  calc_pt(z_d_gen, batch_size=self.batch_size)
        z_d_gen1, z_d_real1, z_d_gen10 = tf.split(self.D_z1, 3)
        self.pulling_term1 =  calc_pt(Flatten()(z_d_gen1), batch_size=self.batch_size)
        z_d_gen2, z_d_real2, z_d_gen20 = tf.split(self.D_z2, 3)
        self.pulling_term2 =  calc_pt(Flatten()(z_d_gen2), batch_size=self.batch_size)
        z_d_gen3, z_d_real3, z_d_gen30 = tf.split(self.D_z3, 3)
        self.pulling_term3 =  calc_pt(Flatten()(z_d_gen3), batch_size=self.batch_size)

        from style import style_loss, total_style_cost
        self.style_loss =  0.5e-13*total_style_cost(tf.transpose(tf.concat([G,G,G],1),(0,2,3,1)),tf.transpose(tf.concat([fix_img,fix_img,fix_img],1),(0,2,3,1)), z_d_gen, self.z, self.batch_size)
        # self.style_loss = tf.Variable(0.)
        # self.style_loss1 = style_loss(tf.sigmoid(z_d_real1),tf.sigmoid(z_d_gen10),self.batch_size,1)
        # self.style_loss2 = style_loss(tf.sigmoid(z_d_real2),tf.sigmoid(z_d_gen20),self.batch_size,1)
        # self.style_loss3 = style_loss(tf.sigmoid(z_d_real3),tf.sigmoid(z_d_gen30),self.batch_size,1)
        # self.style_loss = self.style_loss1*0.3 + self.style_loss2*0.6 + self.style_loss3*2.1  #maybe longer strok

        tmp = ( tf.reduce_mean(G + 1, axis=(1, 2, 3)) - tf.reduce_mean(self.normx + 1, axis=(1, 2, 3)) ) / tf.reduce_mean(self.normx + 1, axis=(1, 2, 3))
        self.d_fvr = tf.reduce_mean(tf.abs(tmp))
        # self.d_fvr = tf.Variable(0.)

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.square(AE_x - self.normx))
        # self.d_loss_fake = tf.reduce_mean(tf.square(AE_G - G))
        self.d_loss_fake = tf.reduce_mean(tf.square(self.normx - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        pt_r = 0.2
        # self.g_loss = self.d_loss_fake*(1+pt_r) + pt_r*self.pulling_term + 0.4*self.style_loss + self.d_fvr
        self.g_loss = self.d_loss_fake + self.style_loss #+ self.d_fvr

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.d_loss_fake
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
            tf.summary.scalar("misc/pt", self.pulling_term),
            tf.summary.scalar("misc/pt1", self.pulling_term1),
            tf.summary.scalar("misc/pt2", self.pulling_term2),
            tf.summary.scalar("misc/pt3", self.pulling_term3),
            tf.summary.scalar("misc/style_loss", self.style_loss),
            tf.summary.scalar("misc/d_fvr", self.d_fvr),
        ])

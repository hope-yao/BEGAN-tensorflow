from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque
from style import total_style_cost, white_style

from models import *
from utils import save_image

def next(loader):
    return loader.next()[0].data.numpy()

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

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
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

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.channel = 1
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        import dateutil.tz
        import datetime
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
        self.logdir, self.modeldir = creat_dir('GAN')
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logdir)

        # # fix y and generate in z plan
        # sess = tf.Session()
        # self.saver.restore(sess, "./models/GAN/GAN_2017_05_31_14_32_26/experiment_155085.ckpt")
        # y_input_fix = np.asarray([[1, 0]] * 16, dtype='float32')
        # z_input_fix = np.asarray([[i / 3., j / 3.] for i in range(4) for j in range(4)], dtype='float32')
        # feed_dict_fix = {self.y: y_input_fix, self.z: z_input_fix}
        # g_img = sess.run(self.G, feed_dict_fix)
        # nrow = 4
        # save_image(g_img, '{}/itr{}.png'.format(self.logdir, 10), nrow=nrow)

        sv = tf.train.Supervisor(logdir=self.model_dir,
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

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False

            self.build_test_model()

    def train(self):
        # z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        # x_fixed = self.get_image_from_loader()
        # save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        def CelebA(datadir, num=200000):
            '''load human face dataset'''
            import h5py
            from random import sample
            import numpy as np
            f = h5py.File("/home/doi5/Documents/Hope/rectcrs_z.hdf5", "r")
            # f = h5py.File(datadir + "/rect_rectcrs0.hdf5", "r")
            data_key = f.keys()[0]
            data = np.asarray(f[data_key], dtype='float32')  # normalized into (-1, 1)
            # data = (np.asarray(f[data_key],dtype='float32') / 255. - 0.5 )*2 # normalized into (-1, 1)
            # data = data.transpose((0,2,3,1))
            label_key = f.keys()[1]
            label = np.asarray(f[label_key])
            z_key = f.keys()[2]
            z = np.asarray(f[z_key])
            z = z.astype('float32') * 2 - 1

            split = 0.1
            l = len(data)  # length of data
            n1 = int(split * l)  # split for testing
            indices = [1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16] * 2#sample(range(l), n1)

            x_test = data[indices]
            y_test = label[indices]
            z_test = z[indices]
            x_train = np.delete(data, indices, 0)
            y_train = np.delete(label, indices, 0)
            z_train = np.delete(z, indices, 0)

            # return (x_train, y_train, 0), (x_test, y_test, 0)
            return (x_train[0:num], y_train[0:num], z_train[0:num]), (x_test[0:1000], y_test[0:1000], z_test[0:1000])

        def Mnist64(datadir, num=200000):
            aa = np.load('/home/doi5/Documents/Hope/Mnist4k_b.npy')
            data = aa.item()['data']
            label = aa.item()['label']
            indices = [1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16]*2#sample(range(l), n1)

            x_test = data[indices]
            y_test = label[indices]
            x_train = np.delete(data, indices, 0)
            y_train = np.delete(label, indices, 0)
            return ( np.expand_dims((x_train+1)*127.5, axis=1), y_train), (np.expand_dims((x_test+1)*127.5, axis=1), y_test)


        counter = 0
        from tqdm import tqdm
        self.datadir='/home/hope-yao/Documents/Data'
        # (self.X_train, self.y_train, self.z_train), (self.X_test, self.y_test, self.z_test) = CelebA(self.datadir)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = Mnist64(self.datadir)
        x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        z_input_fix = np.random.rand(self.batch_size,self.z_num).astype('float32')# temperary, (-1,1)
        feed_dict_fix = {self.x: x_input_fix, self.y: y_input_fix, self.z: z_input_fix}
        for epoch in range(5000):
            it_per_ep = len(self.X_train) / self.batch_size
            for i in tqdm(range(it_per_ep)):
                counter += 1
                x_input = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_input = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                z_input = np.random.rand(self.batch_size, self.z_num).astype('float32')  # temperary
                feed_dict = {self.x: x_input,self.y: y_input, self.z: z_input}
                result = self.sess.run([self.d_loss,self.g_loss,self.measure,self.k_update,self.k_t, self.style_loss, self.pulling_term, self.basis_loss, self.zxz_loss],feed_dict)
                print(result)

                if counter in [5e5, 3e6, 1e7]:
                    self.sess.run([self.g_lr_update, self.d_lr_update])

                if counter % 100 == 0:
                    x_img, x_rec, g_img, g_rec, D0, D1, D2, D3, G0, G1, G2, G3 = \
                        self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G, self.D0, self.D1, self.D2, self.D3, self.G0, self.G1, self.G2, self.G3], feed_dict_fix)
                    nrow = self.batch_size
                    all_G_z = np.concatenate([x_input_fix.transpose((0,2,3,1)), x_rec, g_img, g_rec, D0, D1, D2, D3, G0, G1, G2, G3])
                    save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, counter),nrow=nrow)

                if counter in [1e2, 1e3, 5e3, 1e4, 2e4, 3e4, 1e5, 2e5]:
                    snapshot_name = "%s_%s" % ('experiment', str(counter))
                    fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                    print("Model saved in file: %s" % fn)

                if counter % 10 == 0:
                    summary = self.sess.run(self.summary_op, feed_dict)
                    self.summary_writer.add_summary(summary, counter)
                    self.summary_writer.flush()

                if counter in [2e3, 5e3, 8e3, 11e3, 14e3]:
                    self.sess.run([self.g_lr_update, self.d_lr_update])

                    # for step in trange(self.start_step, self.max_step):
        #     fetch_dict = {
        #         "k_update": self.k_update,
        #         "measure": self.measure,
        #     }
        #     if step % self.log_step == 0:
        #         fetch_dict.update({
        #             "summary": self.summary_op,
        #             "g_loss": self.g_loss,
        #             "d_loss": self.d_loss,
        #             "k_t": self.k_t,
        #         })
        #     result = self.sess.run(fetch_dict)
        #
        #     measure = result['measure']
        #     measure_history.append(measure)
        #
        #     if step % self.log_step == 0:
        #         self.summary_writer.add_summary(result['summary'], step)
        #         self.summary_writer.flush()
        #
        #         g_loss = result['g_loss']
        #         d_loss = result['d_loss']
        #         k_t = result['k_t']
        #
        #         print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
        #               format(step, self.max_step, d_loss, g_loss, measure, k_t))
        #
        #     if step % (self.log_step * 10) == 0:
        #         x_fake = self.generate(z_fixed, self.model_dir, idx=step)
        #         self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)
        #
        #     if step % self.lr_update_step == self.lr_update_step - 1:
        #         self.sess.run([self.g_lr_update, self.d_lr_update])
        #         #cur_measure = np.mean(measure_history)
        #         #if cur_measure > prev_measure * 0.99:
        #         #prev_measure = cur_measure

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, 1, 64, 64])
        x = norm_img(self.x)
        self.x_img = denorm_img(x,self.data_format)
        # self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_num], name='z_input')
        # self.y = tf.cast(tf.multinomial(tf.zeros([self.batch_size, 2]), 2), tf.float32) #second 2 is number of attributes
        self.y = tf.placeholder(tf.float32, [self.batch_size, 4], name='y_input')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_num], name='z_input')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, self.G_var, G0, G1 , G2, G3 = GeneratorCNN(
                self.y, self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)

        d_out, self.D_z, self.D_var, D0, D1 , D2, D3 = DiscriminatorCNN( self.y,
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format)

        z_d_gen, z_d_real = tf.split(self.D_z, 2)
        nom = tf.matmul(z_d_gen, tf.transpose(z_d_gen, perm=[1, 0]))
        denom = tf.sqrt(tf.reduce_sum(tf.square(z_d_gen), reduction_indices=[1], keep_dims=True))
        pt = tf.square(tf.transpose((nom / denom), (1, 0)) / denom)
        pt = pt - tf.diag(tf.diag_part(pt))
        self.pulling_term = tf.reduce_sum(pt) / (self.batch_size * (self.batch_size - 1))

        self.zxz_loss = tf.reduce_mean(tf.abs(self.z - z_d_gen))

        AE_G, AE_x = tf.split(d_out, 2)
        self.G0 = denorm_img(G0, self.data_format)
        self.G1 = denorm_img(G1, self.data_format)
        self.G2 = denorm_img(G2, self.data_format)
        self.G3 = denorm_img(G3, self.data_format)
        self.D0 = denorm_img(D0, self.data_format)
        self.D1 = denorm_img(D1, self.data_format)
        self.D2 = denorm_img(D2, self.data_format)
        self.D3 = denorm_img(D3, self.data_format)

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake + self.zxz_loss
        # self.style_loss, self.sw, self.conv_out2_S, self.conv_out2, self.sl1, self.conv_out4_S, self.conv_out4, self.sl2 =  total_style_cost(tf.transpose(tf.concat([G,G,G],1),(0,2,3,1)),tf.transpose(tf.concat([x,x,x],1),(0,2,3,1)), self.z_gen, self.z)
        # self.style_loss = white_style(tf.transpose(G,(0,2,3,1)))
        self.style_loss = tf.Variable(0.)
        self.basis_loss = tf.reduce_sum(tf.sigmoid(G0)*tf.sigmoid(G1))
        # self.g_loss = tf.reduce_mean(tf.abs(AE_G - G)) + self.style_loss + self.pulling_term + self.g_loss_reg
        self.g_loss = self.d_loss_fake + self.zxz_loss

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
            tf.summary.scalar("misc/style_loss", self.style_loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
            tf.summary.scalar("misc/pulling_term", self.pulling_term),
            tf.summary.scalar("misc/basis_loss", self.basis_loss),
            tf.summary.scalar("misc/zxz_loss", self.zxz_loss),
        ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = GeneratorCNN(
                self.z_r, self.conv_hidden_num, self.channel, self.repeat_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = batch_size/2

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        root_path = "./"#self.model_dir

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            save_image(real1_batch, os.path.join(root_path, 'test{}_real1.png'.format(step)))
            save_image(real2_batch, os.path.join(root_path, 'test{}_real2.png'.format(step)))

            self.autoencode(
                    real1_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real1".format(step)))
            self.autoencode(
                    real2_batch, self.model_dir, idx=os.path.join(root_path, "test{}_real2".format(step)))

            self.interpolate_G(real1_batch, step, root_path)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "test{}_G_z.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
            save_image(all_G_z, '{}/G_z{}.png'.format(root_path, step))

        save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x

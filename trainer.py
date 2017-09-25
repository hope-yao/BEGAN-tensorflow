from __future__ import print_function

from data_loader import *
from models import *
from utils import save_image, list2tensor, creat_dir


# from style import total_style_cost, white_style



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
    image = image / 255.
    # image = image/127.5 - 1.
    image = (image+1)/2 #convert to (0,1)
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc(norm * 255., data_format), 0, 255)
    # return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


class Trainer(object):
    def __init__(self, config):
        self.config = config
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

        # self.gamma = config.gamma
        self.gamma = tf.placeholder(tf.float32,())
        self.lambda_k = config.lambda_k

        self.z_num_sep = config.z_num
        self.z_num = sum(self.z_num_sep)
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        self.imsize = 64
        self.channel = 1
        self.repeat_num = int(np.log2(self.imsize)) - 2 -1

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train

        self.build_model()

        self.logdir, self.modeldir = creat_dir('GAN')
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logdir)

        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # self.saver.restore(self.sess, "./models/GAN/GAN_2017_08_21_14_08_10/experiment_10000.ckpt")
        # self.saver.restore(self.sess, "./models/GAN/GAN_2017_08_12_23_30_24/experiment_102438.ckpt")

    def train(self):

        from tqdm import tqdm
        self.datadir = './data'
        (self.X_train, self.y_train), (self.X_test, self.y_test) = CRS()
        # Mnist128_trans()#CRS()#() Mnist64#
        x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
        mu, sigma = 0, 1  # mean and standard deviation
        z_input_fix = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
        feed_dict_fix = {self.x: x_input_fix, self.y: y_input_fix, self.z: z_input_fix}

        counter = 0
        gamma = 0.5
        for epoch in range(5000):
            it_per_ep = len(self.X_train) / self.batch_size
            for i in tqdm(range(it_per_ep)):
                counter += 1
                x_input = self.X_train[i * self.batch_size:(i + 1) * self.batch_size]
                y_input = self.y_train[i * self.batch_size:(i + 1) * self.batch_size]
                z_input = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
                feed_dict = {self.x: x_input, self.y: y_input, self.z: z_input, self.gamma:gamma}
                result = self.sess.run([self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss, self.k_update,
                                        self.klf_mean, self.klr_mean], feed_dict)
                # drv, dfv, dv, gv, = result[0:4]
                # if gv<2.:
                # # if dfv<1.:
                #     gamma = 1.
                # else:
                #     gamma = 0.5

                print(result)
                for i, val in enumerate(result):
                    if np.any(np.isnan(np.asarray(val))):
                        print('err')

                if counter in [100e3, 200e3]:
                    self.sess.run([self.g_lr_update, self.d_lr_update])

                if counter % 100 == 0:
                    x_img, x_rec, g_img, g_rec, D_sub, G_sub = \
                        self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G, self.D_sub,
                                       self.G_sub], feed_dict_fix)
                    nrow = self.batch_size
                    all_G_z = np.concatenate([x_img, x_rec, g_img, g_rec, D_sub, G_sub])
                    im = save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, counter), nrow=nrow)

                if counter in [10e3, 2e4, 4e4, 6e4, 2e5, 4e5]:
                    snapshot_name = "%s_%s" % ('experiment', str(counter))
                    fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                    print("Model saved in file: %s" % fn)

                if counter % 10 == 0:
                    summary = self.sess.run(self.summary_op, feed_dict)
                    self.summary_writer.add_summary(summary, counter)
                    self.summary_writer.flush()

    def build_model(self):
        n_net = len(self.z_num_sep)
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.channel, self.imsize, self.imsize])
        self.x_norm = x = norm_img(self.x)
        self.y = tf.placeholder(tf.float32, [self.batch_size, n_net], name='y_input')

        mask = []
        for j,_ in enumerate(self.z_num_sep):
            for i in range(n_net):
                mask += [tf.tile(self.y[:, i:i + 1], [1, self.z_num_sep[j]])]
        self.mask = list2tensor(mask, 1)

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_num], name='z_input')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        g_optimizer, d_optimizer = tf.train.AdamOptimizer(self.g_lr), tf.train.AdamOptimizer(self.d_lr)

        tower_grads_d = []
        tower_grads_g = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in [3]:
                with tf.device('/gpu:%d' % i):
                    with tf.variable_scope('G') as vs_g:
                        self.G_norm, self.G_sub_norm = GeneratorCNN(
                            self.y, self.z, self.z_num_sep, self.conv_hidden_num, self.channel,
                            self.repeat_num, self.data_format)
                        self.G_var = tf.contrib.framework.get_variables(vs_g)

                    with tf.variable_scope('D') as vs_d:
                        self.all_x = tf.concat([self.G_norm, x], 0)
                        self.d_z = Encoder(self.all_x, self.z_num, self.repeat_num,
                                           sum(self.conv_hidden_num), self.data_format)
                        L = 1
                        # self.d_zg, z_mean, z_log_var = my_sampling(self.d_z, n_net, L)
                        self.d_out, self.D_sub_norm = Decoder(self.y, self.d_z, self.channel, self.z_num_sep,
                                                                              self.repeat_num,self.conv_hidden_num, self.data_format)
                        self.D_var = tf.contrib.framework.get_variables(vs_d)
                    tf.get_variable_scope().reuse_variables()

                    dz_net0 = tf.slice(self.d_z,(0,0),(2*self.batch_size,self.z_num_sep[0]) )
                    self.pt_z, self.nom, self.denom =  calc_pt_Angular(dz_net0,self.batch_size)
                    self.klf, self.klr, self.mr, self.mf, self.dzf_mean_diag = calc_eclipse_loss_analy(self.d_z,self.z,self.z_num_sep)
                    self.mode_variance = tf.reduce_mean(tf.abs(self.dzf_mean_diag))
                    self.klf_mean = 2*tf.reduce_mean(list2tensor(self.klf))
                    self.klr_mean = 2*tf.reduce_mean(list2tensor(self.klr))
                    self.mr = 2*tf.reduce_mean(list2tensor(self.mr))
                    self.mf = 2*tf.reduce_mean(list2tensor(self.mf))

                    self.AE_G_norm = []
                    self.AE_x_norm = []
                    for i, d_out_l in enumerate(tf.split(self.d_out, L)):
                        AE_G_norm_i, AE_x_norm_i = tf.split(d_out_l, 2)
                        self.AE_G_norm += [AE_G_norm_i]
                        self.AE_x_norm += [AE_x_norm_i]
                    self.d_loss_real = 80  * tf.reduce_mean(tf.square(list2tensor(self.AE_x_norm) - tf.tile(self.x_norm,[L,1,1,1])))
                    self.d_loss_fake = 80 * tf.reduce_mean(tf.square(list2tensor(self.AE_G_norm) - tf.tile(self.G_norm,[L,1,1,1])))

                    # self.part_AE_G_norm = []
                    # self.part_AE_x_norm = []
                    # for i, part_d_out_l in enumerate(tf.split(self.partd_out, L)):
                    #     part_AE_G_norm_i, part_AE_x_norm_i = tf.split(part_d_out_l, 2)
                    #     self.part_AE_G_norm += [part_AE_G_norm_i]
                    #     self.part_AE_x_norm += [part_AE_x_norm_i]
                    # self.part_d_loss_real = 80  * tf.reduce_mean(tf.square(list2tensor(self.part_AE_x_norm) - tf.tile(self.x_norm,[L,1,1,1])))
                    # self.part_d_loss_fake = 80 * tf.reduce_mean(tf.square(list2tensor(self.part_AE_G_norm) - tf.tile(self.G_norm,[L,1,1,1])))

                    # alpha = 0.4
                    # self.mixed_d_loss_real = (1-alpha)*self.d_loss_real + alpha*self.part_d_loss_real
                    # self.mixed_d_loss_fake = (1-alpha)*self.d_loss_fake + alpha*self.part_d_loss_fake

                    self.dgall = tf.zeros((self.batch_size, 1, self.imsize, self.imsize))
                    self.gall = tf.zeros((self.batch_size, 1, self.imsize, self.imsize))
                    for i, dsubnorm_i in enumerate(self.D_sub_norm):
                        img_i, _ = tf.split(dsubnorm_i, 2)
                        self.dgall += img_i
                    for i, img_i in enumerate(self.G_sub_norm):
                        self.gall += img_i
                    self.all_g_loss = 1 * tf.reduce_mean(tf.square(self.dgall - self.gall))

                    self.r_dis = 10*(self.klr_mean + self.mr)
                    self.f_dis = 10*(self.klf_mean + self.mf)
                    self.g_loss = self.d_loss_fake + self.f_dis + self.all_g_loss #+ 20*self.pt_z
                    self.d_loss = self.d_loss_real - self.k_t * self.g_loss + self.r_dis + 5*self.mode_variance
                    # self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake + self.r_dis + self.f_dis #+ (1-self.pt_z)
                    grads_g = g_optimizer.compute_gradients(self.g_loss, var_list=self.G_var)
                    tower_grads_g.append(grads_g)
                    grads_d = d_optimizer.compute_gradients(self.d_loss, var_list=self.D_var)
                    tower_grads_d.append(grads_d)

        from multi_gpu import average_gradients
        mean_grads_g = average_gradients(tower_grads_g)
        mean_grads_d = average_gradients(tower_grads_d)
        apply_gradient_g = g_optimizer.apply_gradients(mean_grads_g, global_step=self.step)
        apply_gradient_d = g_optimizer.apply_gradients(mean_grads_d)

        self.x_img = denorm_img(x, self.data_format)
        self.G = denorm_img(self.G_norm, self.data_format)
        self.AE_G = denorm_img(self.AE_G_norm[0], self.data_format)
        self.AE_x = denorm_img(self.AE_x_norm[0], self.data_format)
        self.G_sub = denorm_img(list2tensor(self.G_sub_norm), self.data_format)
        self.D_sub = denorm_img(list2tensor(self.D_sub_norm), self.data_format)

        self.balance = self.gamma * (self.d_loss_real+self.r_dis) - self.g_loss
        # self.balance = self.gamma * self.d_loss_real - self.d_loss_fake
        self.measure = self.d_loss_real + tf.abs(self.balance)
        with tf.control_dependencies([apply_gradient_d, apply_gradient_g]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),

            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
            tf.summary.scalar("misc/klf_mean", self.klf_mean),
            tf.summary.scalar("misc/klr_mean", self.klr_mean),
            tf.summary.scalar("misc/mr", self.mr),
            tf.summary.scalar("misc/mf", self.mf),
            tf.summary.scalar("misc/pt_z", self.pt_z),
            tf.summary.scalar("misc/all_g_loss", self.all_g_loss),
            tf.summary.scalar("misc/gamma", self.gamma),
            tf.summary.scalar("misc/mode_variance", self.mode_variance),
        ])

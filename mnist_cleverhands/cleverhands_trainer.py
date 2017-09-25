from __future__ import print_function

from data_loader import *
from cleverhands_models import *
from utils import save_image, list2tensor, creat_dir



class Trainer(object):
    def __init__(self, ):

        self.optimizer = 'adam'
        self.batch_size = 64

        self.step = tf.Variable(0, name='step', trainable=False)
        g_lr = 1e-4
        d_lr = 1e-4
        self.g_lr = tf.Variable(g_lr, name='g_lr')
        self.d_lr = tf.Variable(d_lr, name='d_lr')
        self.g_lr_update = tf.assign(self.g_lr, self.g_lr * 0.5, name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, self.d_lr * 0.5, name='d_lr_update')

        # self.gamma = config.gamma
        self.gamma = tf.placeholder(tf.float32,())
        self.lambda_k = 0.01

        self.z_num_sep = [4]*10
        self.z_num = sum(self.z_num_sep)
        self.conv_hidden_num = [16]*10
        self.imsize = 28
        self.channel = 1
        self.repeat_num = int(np.log2(self.imsize)) - 2

        self.build_model()
        tf.flags.DEFINE_string("data_dir", "", "")
        FLAGS = tf.flags.FLAGS
        from tensorflow.examples.tutorials import mnist
        data_directory = os.path.join(FLAGS.data_dir, "mnist")
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        self.train_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # binarized (0-1) mnist data
        self.test_data = mnist.input_data.read_data_sets(data_directory, one_hot=True).train  # binarized (0-1) mnist data
        x_train_fix, y_train_fix = self.train_data.next_batch(self.batch_size)
        self.mu = 0
        self.sigma = 1
        z_input_fix = np.random.normal(self.mu, self.sigma, (self.batch_size, self.z_num)).astype('float32')
        self.feed_dict_fix = {self.x_norm: x_train_fix.reshape((self.batch_size,1,28,28)),
                              self.y: y_train_fix, self.z: z_input_fix, self.gamma:0.5}

        self.logdir, self.modeldir = creat_dir('GAN')
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logdir)

        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def train(self):

        from tqdm import tqdm
        mu, sigma = 0, 1  # mean and standard deviation
        counter = 0
        gamma = 0.5
        for itr in range(50000):
            counter += 1
            x_input, y_input = self.train_data.next_batch(self.batch_size)
            z_input = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
            feed_dict = {self.x_norm: x_input.reshape((self.batch_size,1,28,28)),
                         self.y: y_input, self.z: z_input, self.gamma:gamma}
            result = self.sess.run([self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss, self.k_update,
                                    self.klf_mean, self.klr_mean], feed_dict)
            print(result)
            for i, val in enumerate(result):
                if np.any(np.isnan(np.asarray(val))):
                    print('err')

            if counter in [100e3, 200e3]:
                self.sess.run([self.g_lr_update, self.d_lr_update])

            if counter % 100 == 0:
                x_img, x_rec, g_img, g_rec, D_sub, G_sub = \
                    self.sess.run([self.x_img, self.AE_x, self.G, self.AE_G, self.D_sub,
                                   self.G_sub], self.feed_dict_fix)
                nrow = self.batch_size
                all_G_z = np.concatenate([x_img, x_rec, g_img, g_rec])
                im = save_image(all_G_z.transpose((0,2,3,1)), '{}/itr{}.png'.format(self.logdir, counter), nrow=nrow)

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
        self.x_norm = tf.placeholder(tf.float32, [self.batch_size, self.channel, self.imsize, self.imsize])
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

        self.data_format = 'NCHW'
        with tf.variable_scope(tf.get_variable_scope()):
            gpu_idx = 0
            with tf.device('/gpu:%d' % gpu_idx):
                with tf.variable_scope('G') as vs_g:
                    self.G_norm, self.G_sub_norm = GeneratorCNN(
                        self.y, self.z, self.z_num_sep, self.conv_hidden_num, self.channel,
                        self.repeat_num)
                    self.G_var = tf.contrib.framework.get_variables(vs_g)

                with tf.variable_scope('D') as vs_d:
                    self.all_x = tf.concat([self.G_norm, self.x_norm], 0)
                    self.d_z = Encoder(self.all_x, self.z_num, self.repeat_num,
                                       sum(self.conv_hidden_num))
                    L = 1
                    # self.d_zg, z_mean, z_log_var = my_sampling(self.d_z, n_net, L)
                    self.d_out, self.D_sub_norm = Decoder(self.y, self.d_z, self.channel, self.z_num_sep,
                                                                          self.repeat_num,self.conv_hidden_num)
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

        def denorm_img(norm):
            return tf.clip_by_value(norm * 255., 0, 255)
        self.x_img = denorm_img(self.x_norm)
        self.G = denorm_img(self.G_norm)
        self.AE_G = denorm_img(self.AE_G_norm[0])
        self.AE_x = denorm_img(self.AE_x_norm[0])
        self.G_sub = denorm_img(list2tensor(self.G_sub_norm))
        self.D_sub = denorm_img(list2tensor(self.D_sub_norm))

        self.balance = self.gamma * (self.d_loss_real+self.r_dis) - self.g_loss
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

    def classify(self):
        y_test = [[0]*10]*self.batch_size
        for net_i in range(len(self.z_num_sep)):
            y_test[:,net_i] = 1
            # for z_i in range(100):
            z0 = z1 = z2 = z3 = np.arange(0,1,100)

            G_norm = GeneratorCNN(
                y_test, z, self.z_num_sep, self.conv_hidden_num, self.channel,
                self.repeat_num)

trainer = Trainer()
trainer.train()
print("done")
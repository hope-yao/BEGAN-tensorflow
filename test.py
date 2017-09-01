
# dz = self.sess.run(self.D_z, feed_dict_fix)
#
# x_input_fix_t = x_input_fix
# y_input_fix_t = y_input_fix
# z_input_fix_t = dz[32:64,0:128]
# feed_dict_fix_t = {self.x: x_input_fix_t, self.y: y_input_fix_t, self.z: z_input_fix_t}
#
# gg = self.sess.run([self.G_D,self.G_D0,self.G_D1,self.G_D2,self.G_D3], feed_dict_fix_t)
#
# all_G_z = np.concatenate([gg[0],gg[1],gg[2],gg[3],gg[4]])
# im = save_image(all_G_z, '{}/itr{}.png'.format(self.logdir, 0), nrow=nrow)


# np.mean(dz[32:64,0:32]),np.mean(dz[32:64,32:64]),np.mean(dz[32:64,64:96]),np.mean(dz[32:64,98:128])
# dz[0:32,0:32],dz[0:32,32:64],dz[0:32,64:96],dz[0:32,98:128]
#
#
#
#     dzr0 = dz[32:64,0:32]
#     array([ 0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.,
#             0.,  1.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  1.,  1.,  0.,  1.,
#             0.,  0.,  0.,  0.,  1.,  1.])

def z_tsne2():
    from sklearn.manifold import TSNE
    z1 = z[:,8:]
    dzf1 = dz[:64,8:]
    dzr1 = dz[64:,8:]
    z0 = z[:,:8]
    dzf0 = dz[:64,:8]
    dzr0 = dz[64:,:8]
    X = np.concatenate([z1,dzr1,dzf1])
    model = TSNE(n_components=2, perplexity=10.0, random_state=0)
    np.set_printoptions(suppress=True)
    T = model.fit_transform(X)
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    for i,k in enumerate(T):
        if i<64*1:
            plt.plot(T[i,0],T[i,1],'k.')
        elif i<64*2:
            plt.plot(T[i,0],T[i,1],'r.')
        elif i<64*3:
            plt.plot(T[i,0],T[i,1],'b.')

#########################################
def z_tsne():
    z_num = 32
    bs = 32
    dz0_f = dz[0:bs,0:z_num]
    dz1_f = dz[0:bs,z_num:z_num*2]
    dz0_r = dz[bs:bs*2,0:z_num]
    dz1_r = dz[bs:bs*2,z_num:z_num*2]
    z0_r = z[:,0:z_num]
    z1_r = z[:,z_num:z_num*2]

    X=np.concatenate([z1_r,dz1_f,dz1_r],0)
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, perplexity=15.0, random_state=0)
    np.set_printoptions(suppress=True)
    T = model.fit_transform(X)

    import seaborn as sns
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # ax = plt.axes(projection='3d')
    for i,k in enumerate(T):
        if i<32:
            plt.plot(T[i,0],T[i,1],'bo',)
    #         ax.scatter(T[i,0],T[i,1],T[i,2],c='b')
        elif i<64:
            plt.plot(T[i,0],T[i,1],'ro')
    #         ax.scatter(T[i,0],T[i,1],T[i,2],c='r')
        else:
            plt.plot(T[i,0],T[i,1],'ko')
    plt.show()



def z_dist_viz():
    import numpy as np
    import seaborn as sns

    y=np.load('./testing/y.npy')
    z=np.load('./testing/z.npy')
    dz=np.load('./testing/dz.npy')

    z_num=32
    bs=32
    mask  =np.concatenate([ np.tile(y[:,0:1],(1,32)),np.tile(y[:,0:1],(1,32)),np.tile(y[:,0:1],(1,32)),np.tile(y[:,0:1],(1,32))],1)
    dzr=dz[bs:bs*2,:]
    dzf=dz[0:bs,:]

    dzr2=dzr[:,0:z_num]
    dzr2 = dzr2[mask[:,0:z_num]!= 0]
    dzr4=dzr[:,z_num:z_num*2]
    dzr4 = dzr4[mask[:,z_num:z_num*4]!= 0]
    dzr6=dzr[:,z_num*2:z_num*3]
    dzr6 = dzr6[mask[:,z_num*2:z_num*3]!= 0]
    dzr8=dzr[:,z_num*3:z_num*4]
    dzr8 = dzr8[mask[:,z_num*3:z_num*4]!= 0]

    dzf2=dzf[:,0:z_num]
    dzf2 = dzf2[mask[:,0:z_num]!= 0]
    dzf4=dzf[:,z_num:z_num*2]
    dzf4 = dzf4[mask[:,z_num:z_num*2]!= 0]
    dzf6=dzf[:,z_num*2:z_num*3]
    dzf6 = dzf6[mask[:,z_num*2:z_num*3]!= 0]
    dzf8=dzf[:,z_num*3:z_num*4]
    dzf8 = dzf8[mask[:,z_num*3:z_num*4]!= 0]

    z2=z[:,0:z_num]
    z2 = z2[mask[:,0:z_num]!= 0]
    z4=z[:,z_num:z_num*2]
    z4 = z4[mask[:,z_num:z_num*2]!= 0]
    z6=z[:,z_num*2:z_num*3]
    z6 = z6[mask[:,z_num*2:z_num*3]!= 0]
    z8=z[:,z_num*3:z_num*4]
    z8 = z8[mask[:,z_num*3:z_num*4]!= 0]



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(5,5))
    gs1 = gridspec.GridSpec(2, 2)


    ax2 = plt.subplot(gs1[0:1, 0:1])
    sns.distplot(dzr2.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax2)
    sns.distplot(dzf2.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax2)
    sns.distplot(z2.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax2)
    plt.setp(ax2, title='digit2')

    ax4 = plt.subplot(gs1[0:1, 1:2])
    sns.distplot(dzr4.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax4)
    sns.distplot(dzf4.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax4)
    sns.distplot(z4.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax4)
    plt.setp(ax4, title='digit4')

    ax6 = plt.subplot(gs1[1:2, 0:1])
    sns.distplot(dzr6.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax6)
    sns.distplot(dzf6.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax6)
    sns.distplot(z6.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax6)
    plt.setp(ax6, title='digit6')

    ax8 = plt.subplot(gs1[1:2, 1:2])
    sns.distplot(dzr8.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax8)
    sns.distplot(dzf8.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax8)
    sns.distplot(z8.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax8)
    plt.setp(ax8, title='digit8')

    plt.show()




def save_var(self):
    y = self.sess.run(self.y,feed_dict_fix)
    z = self.sess.run(self.z,feed_dict_fix)
    dz = self.sess.run(self.d_z,feed_dict_fix)

    np.save('./testing/y.npy',y)
    np.save('./testing/z.npy',z)
    np.save('./testing/dz.npy',dz)

############################################
import numpy as np
import seaborn as sns

def vis_z_dist():
    y=np.load('y.npy')
    z=np.load('z.npy')
    dz=np.load('dz.npy')

    mask  = np.concatenate([ np.tile(y[:,0:1],(1,8)),np.tile(y[:,1:2],(1,8))],1)

    dzr=dz[32:64,:]
    dzf=dz[0:32,:]

    dzr2=dzr[:,0:8]
    dzr2 = dzr2[mask[:,0:8]!= 0]
    dzr4=dzr[:,8:16]
    dzr4 = dzr4[mask[:,8:16]!= 0]


    dzf2=dzf[:,0:8]
    dzf2 = dzf2[mask[:,0:8]!= 0]
    dzf4=dzf[:,8:16]
    dzf4 = dzf4[mask[:,8:16]!= 0]

    z2=z[:,0:8]
    z2 = z2[mask[:,0:8]!= 0]
    z4=z[:,8:16]
    z4 = z4[mask[:,8:16]!= 0]



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    plt.figure(figsize=(5,5))
    gs1 = gridspec.GridSpec(2, 2)


    ax2 = plt.subplot(gs1[0:1, 0:1])
    sns.distplot(dzr2.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax2)
    sns.distplot(dzf2.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax2)
    sns.distplot(z2.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax2)
    plt.setp(ax2, title='digit2')

    ax4 = plt.subplot(gs1[0:1, 1:2])
    sns.distplot(dzr2.flatten(),kde_kws={"color": "k", "lw": 3, "label": "dz_real"}, ax = ax4)
    sns.distplot(dzf2.flatten(),kde_kws={"color": "r", "lw": 3, "label": "dz_fake"}, ax = ax4)
    sns.distplot(z4.flatten(),kde_kws={"color": "b", "lw": 3, "label": "z_input"}, ax = ax4)
    plt.setp(ax4, title='digit4')

    plt.show()


########################################################################################
def interp(self):
    zz0 = np.asarray([z1_r[0]*i/31. + dz1_f[0] *(31-i)/31. for i in range(32)],dtype='float32')
    zz1 = np.asarray([z1_r[0]*i/31. + dz1_f[0] *(31-i)/31. for i in range(32)],dtype='float32')
    zz = np.concatenate([zz0,zz1],1)

    feed_dict_fix0 = {self.x: x_input_fix, self.y: y_input_fix, self.z: zz}

    G_sub = self.sess.run(self.G_sub, feed_dict_fix0)
    nrow = self.batch_size
    all_G_z = np.concatenate([x_img, x_rec, g_img, g_rec, D_sub, G_sub])
    im = save_image(G_sub, '{}/itr{}.png'.format(self.logdir, 1),nrow=nrow)



def norm_img_vis():
    self.datadir='/home/hope-yao/Documents/Data'
    (self.X_train, self.y_train), (self.X_test, self.y_test) = CRS()
    #Mnist128_trans()#Mnist64_switch()#Mnist64() #
    x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
    y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
    mu, sigma = 0, 1  # mean and standard deviation
    z_input_fix = np.random.normal(mu, sigma,(self.batch_size,self.z_num)).astype('float32')
    feed_dict_fix = {self.x: x_input_fix, self.y: y_input_fix, self.z: z_input_fix}

    G_norm,G_sub_norm=self.sess.run([self.G_norm,self.G_sub_norm], feed_dict_fix)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    gsubnorm=np.load('gsubnorm.npy')
    plt.imshow(gsubnorm[3,2,0,:,:])
    plt.colorbar()
    plt.show()

    gsub=np.load('gsub.npy')
    plt.imshow(gsub[0+32*3,:,:,0])
    plt.colorbar()
    plt.show()

    gnorm=np.load('gnorm.npy')
    plt.imshow(gnorm[2,0,:,:])
    plt.colorbar()
    plt.show()

    g=np.load('g.npy')
    plt.imshow(gnorm[2,:,:,0])
    plt.colorbar()
    plt.show()

















dz = self.sess.run(self.d_z, feed_dict_fix)
z = self.sess.run(self.z, feed_dict_fix)
np.save('dz.npy',dz)
np.save('z.npy',z)
np.save('y.npy',y_input)

idx = 0
dzf_v = dz[idx]
dzr_v = dz[idx+32]
z_v = z[idx]
########################################################################################

idx1 = 0
idx2 = 2
zz1 = dz[idx1]
zz2 = dz[idx2]
z_list1 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
zz1 = dz[idx1+32]
zz2 = dz[idx2+32]
z_list2 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
zz = np.asarray(z_list1+z_list2,dtype='float32')

feed_dict_fix0 = {self.y: y_input_fix, self.d_z: zz}#np.ones((32,4))

g_rec, G_sub = self.sess.run([self.AE_x,self.D_sub], feed_dict_fix0)

nrow = self.batch_size
all_G_z = np.concatenate([g_rec, G_sub ])

im = save_image(G_sub , '{}/itr{}.png'.format(self.logdir, 1),nrow=nrow)

########################################################################################

idx1 = 3
idx2 = 2
zz1 = dz[idx1]
zz2 = z[idx2]
z_list1 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
zz1 = dz[idx1+32]
zz2 = z[idx2]
z_list2 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
zz = np.asarray(z_list1+z_list2,dtype='float32')

feed_dict_fix0 = {self.y: y_input_fix, self.d_z: zz}#np.ones((32,4))

g_rec, G_sub = self.sess.run([self.AE_x,self.D_sub], feed_dict_fix0)

nrow = self.batch_size
all_G_z = np.concatenate([g_rec, G_sub ])

im = save_image(G_sub , '{}/itr{}.png'.format(self.logdir, 0),nrow=nrow)


########################################################################################

idx1 = 3
idx2 = 2
zz1 = dz[idx1+32]
zz2 = z[idx2]
z_list1 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
# zz1 = dz[idx1+32]
# zz2 = z[idx2]
# z_list2 = [ zz2 *(31-i)/31. + zz1*i/31.  for i in range(32)]
zz = np.asarray(z_list1,dtype='float32')

feed_dict_fix0 = {self.y: y_input_fix, self.z: zz}#np.ones((32,4))

G_sub = self.sess.run(self.G_sub, feed_dict_fix0)

nrow = self.batch_size

im = save_image(G_sub , '{}/itr{}.png'.format(self.logdir, 0),nrow=nrow)




########################################################################################
import tensorflow as tf

self.sess=tf.Session()
self.saver = tf.train.import_meta_graph('./models/GAN/GAN_2017_07_30_15_13_16/experiment_385977.ckpt.meta')
FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
self.sess = tf.Session(config=tfconfig)
self.saver.restore(self.sess,tf.train.latest_checkpoint('./models/GAN/GAN_2017_07_30_15_13_16'))

self.datadir = '/home/hope-yao/Documents/Data'
(self.X_train, self.y_train), (self.X_test, self.y_test) = Mnist64_switch()
# Mnist128_trans()#CRS()#() Mnist64#
x_input_fix = self.X_test[0 * self.batch_size:(0 + 1) * self.batch_size]
y_input_fix = self.y_test[0 * self.batch_size:(0 + 1) * self.batch_size]
mu, sigma = 0, 1  # mean and standard deviation
z_input_fix = np.random.normal(mu, sigma, (self.batch_size, self.z_num)).astype('float32')
feed_dict_fix = {self.x: x_input_fix, self.y: y_input_fix, self.z: z_input_fix}
# result = self.sess.run([self.d_loss_real, self.d_loss_fake, self.d_loss, self.g_loss], feed_dict_fix)




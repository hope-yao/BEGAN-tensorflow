#
# import tensorflow as tf
#
# slim = tf.contrib.slim
# import numpy as np
#
# fc_size = 12 * 2
#
#
import numpy as np
import tensorflow as tf
from trainer import norm_img, denorm_img, CelebA
from models import reshape, upscale, GeneratorCNN, DiscriminatorCNN
batch_size = 16
data_format = 'NCHW'
z_num = 64
channel = 1
repeat_num = 4
conv_hidden_num = 32

x = tf.placeholder(tf.float32, [batch_size, 1, 192, 192])
normx = norm_img(x)
x_img = denorm_img(x,data_format)
AE_x, D_z, D_var = DiscriminatorCNN(normx, channel, z_num, repeat_num, conv_hidden_num, data_format)
AE_x_img = denorm_img(AE_x,data_format)
#
# d_loss = tf.reduce_mean(tf.square(AE_x - normx))
# d_optimizer = tf.train.AdamOptimizer(1e-3)
# d_optim = d_optimizer.minimize(d_loss, var_list=D_var)
#
# from tqdm import tqdm
from utils import save_image
from trainer import creat_dir
logdir, modeldir = creat_dir('GAN')
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
(X_train, y_train), (X_test, y_test) = CelebA('/home/doi5/Documents/Hope')
x_input_fix = X_test[0 * batch_size:(0 + 1) * batch_size]
feed_dict_fix = {x: x_input_fix}
# counter = 0
# for epoch in range(5000):
#     it_per_ep = len(X_train) / batch_size
#     for i in tqdm(range(it_per_ep)):
#         counter += 1
#         x_input = X_train[i * batch_size:(i + 1) * batch_size]
#         feed_dict = {x: x_input }
#         result = sess.run([d_loss, d_optimizer],feed_dict)
#         print(result[0])
#
#         if counter % 10 == 0:
#             xx_img, x_rec = \
#                 sess.run([x_img, AE_x_img], feed_dict_fix)
#             nrow = 16
#             all_G_z = np.concatenate([x_input_fix.transpose((0,2,3,1)), x_rec])
#             save_image(all_G_z, '{}/itr{}.png'.format(logdir, counter),nrow=nrow)
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(AE_x - normx , 2))
# optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cost)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    count = 0
    for epoch in range(100):
        # Loop over all batches
        for i in range(60):
            batch_xs = X_train[i * batch_size:(i + 1) * batch_size]
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})
            print('iteration:{} loss:{}'.format(count,c))
            # Display logs per epoch step
            if count % 10 == 0:
                xx_img, x_rec = \
                    sess.run([x_img, AE_x_img], feed_dict_fix)
                nrow = 16
                all_G_z = np.concatenate([x_input_fix.transpose((0,2,3,1)), x_rec])
                save_image(all_G_z, '{}/itr{}.png'.format(logdir, count),nrow=nrow)
            count += 1
    print("Optimization Finished!")


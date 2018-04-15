############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

def build_BEGAN_model(self):
    self.x = self.data_loader
    x = norm_img(self.x)

    self.z = tf.random_uniform(
            (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
    self.k_t = tf.Variable(0., trainable=False, name='k_t')

    G, self.G_var = GeneratorCNN(
            self.z, self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse=False)

    d_out, self.D_z, self.D_var = DiscriminatorCNN(
            tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
            self.conv_hidden_num, self.data_format)
    AE_G, AE_x = tf.split(d_out, 2)

    self.G = denorm_img(G, self.data_format)
    self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

    if self.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer
    else:
        raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

    g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

    self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
    self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

    self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
    self.g_loss = d_loss_fake

    d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
    g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

    self.balance = self.gamma * self.d_loss_real - self.g_loss
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
    ])
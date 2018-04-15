############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

from architectures import *
from ops import *
from utils import *

def build_BEGAN_model(self,start_step=0,d_lr=0.00008,g_lr=0.00008,
    gamma=0.5,
    lambda_k=0.001,
    debug=False):
    real_A = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.input_channels_A ],
                                    name='input_images_of_real_samples')
    # B contains Hyperspectral Images
    hsi_B = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.hsi_channels_B ],
                                    name='input_images_of_hs_samples')
    real_B = self.compressor(hsi_B,output_dim=self.input_channels_B) 
    k_t_A = tf.Variable(0., trainable=False, name='k_t_A')
    k_t_B = tf.Variable(0., trainable=False, name='k_t_B')
    step = tf.Variable(start_step, trainable=False, name='global_step')
###  define graphs
    fake_A = self.A_g_net(real_B, reuse = False)
    fake_B = self.B_g_net(real_A, reuse = False)

    AE_restore_real_A = D_AE(real_A, name='A_d', reuse = False)
    real_A_loss = tf.reduce_mean(tf.abs(real_A-AE_restore_real_A))

    AE_restore_fake_A = D_AE(fake_A, name='A_d', reuse = True)
    fake_A_loss = tf.reduce_mean(tf.abs(fake_A-AE_restore_fake_A))

    AE_restore_real_B = D_AE(real_B, name='B_d', reuse = False)
    real_B_loss = tf.reduce_mean(tf.abs(real_B-AE_restore_real_B))

    AE_restore_fake_B = D_AE(fake_B, name='B_d', reuse = True)
    fake_B_loss = tf.reduce_mean(tf.abs(fake_B-AE_restore_fake_B))

### discriminator loss    
    Ad_loss = real_A_loss - k_t_A*fake_A_loss
    Bd_loss = real_B_loss - k_t_B*fake_B_loss
### generator adversarial loss
    Ag_adv_loss = fake_A_loss
    Bg_adv_loss = fake_B_loss
### generator cycle loss
    recover_A = self.A_g_net(fake_B, reuse = True)
    recover_B = self.B_g_net(fake_A, reuse = True)
    cycle_loss_A =  tf.reduce_mean(tf.abs(recover_A - real_A))
    cycle_loss_B = tf.reduce_mean(tf.abs(recover_B - real_B))
### structure loss
    hsi_gx2,hsi_gy2,hsi_gxy =hs_structures= structure_tensor(hsi_B,name='spectral',reuse=False)
    cmp_gx2,cmp_gy2,cmp_gxy =rgb_strucures= structure_tensor(fake_A,name='pseudo_rgb',reuse=False)
    structure_loss = sum([tf.reduce_mean(tf.abs(hs-rgb)) \
                        for hs,rgb in zip(hs_structures,rgb_strucures)])
### overall loss
    d_loss = Ad_loss + Bd_loss # discriminator overall loss
    A_loss = Ad_loss+Ag_adv_loss+cycle_loss_A # Ad Ag loss
    B_loss = Bd_loss+Bg_adv_loss+cycle_loss_B # Bd Bg loss
    g_loss = Ag_adv_loss + Bg_adv_loss \
            + self.lambda_S*structure_loss \
            + self.lambda_A*cycle_loss_A \
            + self.lambda_B*cycle_loss_B
### bind names for "run_cyclegan_optim()"'s use
    A_d_loss_fake = fake_A_loss
    A_d_loss_real = real_A_loss
    B_d_loss_fake = fake_B_loss
    B_d_loss_real = real_B_loss
    A_d_loss = Ad_loss
    B_d_loss = Bd_loss
    translated_A = fake_B
    translated_B = fake_A
    A_g_loss = Ag_adv_loss + self.lambda_A*cycle_loss_A
    B_g_loss = Bg_adv_loss + self.lambda_B*cycle_loss_B
### define trainable variables
    t_vars = tf.trainable_variables()

    Ad_vars = [var for var in t_vars if 'A_d' in var.name]        
    Bd_vars = [var for var in t_vars if 'B_d' in var.name]        
    Ag_vars = [var for var in t_vars if 'A_g' in var.name]
    Bg_vars = [var for var in t_vars if 'B_g' in var.name]
    C_vars = [var for var in t_vars if 'C_' in var.name]
    d_vars = Ad_vars + Bd_vars
    g_vars = Ag_vars + Bg_vars + C_vars

### optimize
    optimizer=tf.train.AdamOptimizer
    g_optimizer, d_optimizer = optimizer(g_lr), optimizer(d_lr)
    d_optim = d_optimizer.minimize(d_loss, var_list=d_vars)
    g_optim = g_optimizer.minimize(g_loss, global_step=step, var_list=g_vars)

### update k
    # if fake_D_loss is too high,  balance<0, kt becomes smaller,
    #    then, AE is trained to reconstruct real image better.
    # if fake_D_loss is too low, balance>0, kt becomes larger,
    #    then, AE is trained to reconstruct fake image worse.
    balance_A = gamma * real_A_loss - fake_A_loss
    measure_A = real_A_loss + tf.abs(balance_A)
    balance_B = gamma * real_B_loss - fake_B_loss
    measure_B = real_B_loss + tf.abs(balance_B)
    with tf.control_dependencies([d_optim, g_optim]):
        self.k_A_update = tf.assign(
            k_t_A, tf.clip_by_value(k_t_A + lambda_k * balance_A, 0, 1))
        self.k_B_update = tf.assign(
            k_t_B, tf.clip_by_value(k_t_B + lambda_k * balance_B, 0, 1))

### Summaries
    ### define summary scalar
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    Ag_adv_loss_sum = tf.summary.scalar("Ag_adv_loss", Ag_adv_loss)
    Bg_adv_loss_sum = tf.summary.scalar("Bg_adv_loss", Bg_adv_loss)
    structure_loss_sum = tf.summary.scalar("structure_loss", structure_loss)
    k_A_sum = tf.summary.scalar("k_t_A", k_t_A)
    k_B__sum = tf.summary.scalar("k_t_B", k_t_B)
    balance_A_sum = tf.summary.scalar("balance_A", balance_A)
    balance_B_sum = tf.summary.scalar("balance_B", balance_B)
    measure_A_sum = tf.summary.scalar("measure_A", measure_A)
    measure_B_sum = tf.summary.scalar("measure_B", measure_B)
    ### define summary image
    real_A_summary_image = tf.summary.image('real_A', real_A)
    fake_A_summary_image = tf.summary.image('fake_A', fake_A)
    real_B_summary_image = tf.summary.image('real_B', real_B)
    fake_B_summary_image = tf.summary.image('fake_B', fake_B)
    all_summaries = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)
### Allocate GPU memory. because queue initializer will run them right after.
    if debug:
        print('initializer received debug option')
    else:
        self.sess.run(tf.global_variables_initializer()) 
        print('Memory allocated for all variables on GPU.')
### bind variables to global scope
    self.__dict__.update(locals())

################
### test uses cyclegan test function
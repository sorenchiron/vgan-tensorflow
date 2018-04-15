############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

import scipy.misc as sm
from architectures import *
from ops import *
from utils import *

def build_fully_cyclegan_model(self):
    real_A = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.input_channels_A ],
                                    name='input_images_of_A_network')
    # B contains Hyperspectral Images
    real_B =hsi_B= tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.hsi_channels_B ],
                                    name='input_images_of_B_network')
    # 3 channel HSI
    false_color_B = self.compressor(real_B,output_dim=self.input_channels_B)

### define graphs
    fake_false_color_B = self.B_g_net(real_A, reuse = False)
    fake_B = simple_extender(fake_false_color_B, output_dim=self.hsi_channels_B,reuse=False)
    BD_predicts_fake = self.B_d_net(fake_B, reuse = False)

    fake_A = self.A_g_net(false_color_B, reuse = False)
    AD_predicts_fake = self.A_d_net(fake_A, reuse = False)
    # compability hooks
    translated_B = fake_A
    translated_A = fake_B
### define structure loss
    hsi_gx2,hsi_gy2,hsi_gxy =hs_structures= structure_tensor(real_B,name='spectral',reuse=False)
    cmp_gx2,cmp_gy2,cmp_gxy =rgb_strucures= structure_tensor(real_B,name='pseudo_rgb',reuse=False)
    structure_loss = sum([tf.reduce_mean(tf.abs(hs-rgb)) \
                        for hs,rgb in zip(hs_structures,rgb_strucures)])
### variance loss
    #variance_loss =  - tf.reduce_mean(tf.pow(real_B - tf.reduce_mean(real_B),2))
### define loss
    recover_A = self.A_g_net(self.compressor(fake_B,output_dim=self.input_channels_B,reuse=True), reuse = True)
    recover_B = simple_extender(self.B_g_net(fake_A, reuse = True),output_dim=self.hsi_channels_B,reuse=True)
### cycle loss
    if self.loss_metric == 'L1':
        A_loss = tf.reduce_mean(tf.abs(recover_A - real_A))
        B_loss = tf.reduce_mean(tf.abs(recover_B - real_B))
    elif self.loss_metric == 'L2':
        A_loss = tf.reduce_mean(tf.square(recover_A - real_A))
        B_loss = tf.reduce_mean(tf.square(recover_B - real_B))
### B discriminator loss
    BD_predicts_real = self.B_d_net(real_B, reuse = True)
    B_d_loss_real = sum_cross_entropy(BD_predicts_real, tf.ones_like(BD_predicts_real))
    B_d_loss_fake = sum_cross_entropy(BD_predicts_fake, tf.zeros_like(BD_predicts_fake)) 
    B_d_loss = B_d_loss_fake + B_d_loss_real
    B_d_inverse_loss = sum_cross_entropy(BD_predicts_real, tf.zeros_like(BD_predicts_real))
### B generator loss (adv loss + cycle loss + structure loss)
    B_g_loss = sum_cross_entropy(BD_predicts_fake, tf.ones_like(BD_predicts_fake)) + self.lambda_B * (B_loss )
### A discriminator loss    
    AD_predicts_real = self.A_d_net(real_A, reuse = True)
    A_d_loss_real = sum_cross_entropy(AD_predicts_real, tf.ones_like(AD_predicts_real))
    A_d_loss_fake = sum_cross_entropy(AD_predicts_fake, tf.zeros_like(AD_predicts_fake)) 
    A_d_loss = A_d_loss_fake + A_d_loss_real
### A generator loss
    A_g_loss = sum_cross_entropy(AD_predicts_fake, tf.ones_like(AD_predicts_fake)) + self.lambda_A * (A_loss )
    
    d_loss = A_d_loss + B_d_loss
    g_loss = A_g_loss + B_g_loss #+ self.lambda_S*structure_loss + self.lambda_V*variance_loss
### compressor loss
    ### inverse loss
    C_loss = sum_cross_entropy(B_d_loss_real, tf.zeros_like(B_d_loss_real))
    ### define summary scalar
    A_d_loss_sum = tf.summary.scalar("A_d_loss", A_d_loss)
    A_loss_sum = tf.summary.scalar("A_loss", A_loss)
    B_d_loss_sum = tf.summary.scalar("B_d_loss", B_d_loss)
    B_loss_sum = tf.summary.scalar("B_loss", B_loss)
    A_g_loss_sum = tf.summary.scalar("A_g_loss", A_g_loss)
    B_g_loss_sum = tf.summary.scalar("B_g_loss", B_g_loss)
    C_loss_sum = tf.summary.scalar('C_loss',C_loss)
    #structure_loss_sum = tf.summary.scalar("structure_loss", structure_loss)
    ### define summary histogram
    BD_predicts_real_hist = tf.summary.histogram('BD_predicts_real_hist', BD_predicts_real)
    BD_predicts_fake_hist = tf.summary.histogram('BD_predicts_fake_hist', BD_predicts_fake)
    AD_predicts_real_hist = tf.summary.histogram('AD_predicts_real_hist', AD_predicts_real)
    AD_predicts_fake_hist = tf.summary.histogram('AD_predicts_fake_hist', AD_predicts_fake)
    ### define summary image
    real_A_summary_image = tf.summary.image('real_A', real_A)
    real_B_summary_image = tf.summary.image('false_color_B', false_color_B)
    translated_A_summary_image = tf.summary.image('fake_false_color_B', fake_false_color_B)
    translated_B_summary_image = tf.summary.image('fake_A', fake_A)
    recover_A_summary_image = tf.summary.image('recover_A', recover_A)
    #recover_B_summary_image = tf.summary.image('recover_B', recover_B)
    all_summaries = tf.summary.merge_all()

    ## define trainable variables
    t_vars = tf.trainable_variables()

    A_d_vars = [var for var in t_vars if 'A_d' in var.name]
    B_d_vars = [var for var in t_vars if 'B_d' in var.name]
    
    A_g_vars = [var for var in t_vars if 'A_g' in var.name]
    B_g_vars = [var for var in t_vars if 'B_g' in var.name]
    
    c_vars = [var for var in t_vars if 'C_' in var.name]
    e_vars = [var for var in t_vars if 'E_' in var.name]

    d_vars = A_d_vars + B_d_vars 
    g_vars = A_g_vars + B_g_vars + c_vars + e_vars

    ga_opt_vars = A_g_vars + c_vars + e_vars
    
    gb_opt_vars = B_g_vars

    saver = tf.train.Saver(max_to_keep=2)

    self.sess.run(tf.global_variables_initializer()) # pre-initialize here so that we can manipulate obj directly
    print('Memory allocated for all variables on GPU.')
    self.__dict__.update(locals())

def fully_cyclegan_test(self,args):
    if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
        
    else:
        print(" [error] Load failed...")
        print(" aborted...")
        return
        
    dcmall_registered = sm.imread('datasets/hsimg/registered_DC_large.bmp')

    rgb_slicer = BigImagePuzzler(dcmall_registered,crop_size=self.image_size,normalize='rgb')
    hs_slicer = BigImagePuzzler(self.hs_flow.imgs[0],crop_size=self.image_size,normalize=args.normalize)
    rgb_puzzles = rgb_slicer.slice()
    hs_puzzles = hs_slicer.slice()
    print('start running')
    fake_a,recover_a,false_color_b,fake_false_color_b=self.sess.run([self.fake_A,
        self.recover_A,
        self.false_color_B,
        self.fake_false_color_B],
        feed_dict={self.real_A:rgb_puzzles,self.hsi_B:hs_puzzles})
    print('recovering')
    img_fake_a = rgb_slicer.recover(fake_a,scale=True)
    img_recover_a = rgb_slicer.recover(recover_a,scale=True)
    img_false_color_b = rgb_slicer.recover(false_color_b,scale=True)
    img_fake_false_color_b = rgb_slicer.recover(fake_false_color_b,scale=True)

    force_exist(os.path.join('test',self.dir_name))

    sm.imsave(os.path.join('test',self.dir_name,'img_fake_a.png'),img_fake_a)
    sm.imsave(os.path.join('test',self.dir_name,'img_recover_a.png'),img_recover_a)
    sm.imsave(os.path.join('test',self.dir_name,'img_false_color_b.png'),img_false_color_b)
    sm.imsave(os.path.join('test',self.dir_name,'img_fake_false_color_b.png'),img_fake_false_color_b)
    print('done')
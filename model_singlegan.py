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

def build_singlegan_model(self):
    self.real_A = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.input_channels_A ],
                                    name='input_images_of_real_samples')
    # B contains Hyperspectral Images
    self.hsi_B = tf.placeholder(tf.float32,
                                    [None, self.image_size, self.image_size,
                                     self.hsi_channels_B ],
                                    name='input_images_of_hs_samples')
###  define graphs
    self.translated_B = self.A_g_net(self.hsi_B, reuse = False)
    self.fake_A = self.translated_B
    self.AD_predicts_real = self.A_d_net(self.real_A, reuse = False)
    self.AD_predicts_fake = self.A_d_net(self.fake_A, reuse = True)
    self.real_A_contour = contour(self.real_A, reuse=False)
    self.fake_A_contour = contour(self.translated_B, reuse=True)
    self.ADC_predicts_real = self.A_d_net(self.real_A_contour,prefix='A_d_C',reuse=False)
    self.ADC_predicts_fake = self.A_d_net(self.fake_A_contour,prefix='A_d_C',reuse=True)

### A discriminator loss    
    self.A_d_loss_real = sum_cross_entropy(self.AD_predicts_real, tf.ones_like(self.AD_predicts_real))
    self.A_d_loss_fake = sum_cross_entropy(self.AD_predicts_fake, tf.zeros_like(self.AD_predicts_fake)) 
    self.A_d_loss = self.A_d_loss_fake + self.A_d_loss_real
### A contour discriminator loss
    self.A_d_C_loss_real = sum_cross_entropy(self.ADC_predicts_real, tf.ones_like(self.ADC_predicts_real))
    self.A_d_C_loss_fake = sum_cross_entropy(self.ADC_predicts_fake, tf.zeros_like(self.ADC_predicts_fake)) 
    self.A_d_C_loss = self.A_d_loss_fake + self.A_d_loss_real
### A generator loss
    self.A_g_loss = sum_cross_entropy(self.AD_predicts_fake, tf.ones_like(self.AD_predicts_fake)) + \
                    sum_cross_entropy(self.ADC_predicts_fake, tf.ones_like(self.ADC_predicts_fake)) 
    self.A_loss = self.A_d_loss + self.lambda_C*self.A_d_C_loss + self.A_g_loss
    self.d_loss = self.A_d_loss + self.lambda_C*self.A_d_C_loss
    self.g_loss = self.A_g_loss

### Summaries
    ### define summary scalar
    self.A_d_loss_sum = tf.summary.scalar("A_d_loss", self.A_d_loss)
    self.A_d_C_loss_sum = tf.summary.scalar("A_d_C_loss", self.A_d_C_loss)
    self.A_loss_sum = tf.summary.scalar("A_loss", self.A_loss)
    self.A_g_loss_sum = tf.summary.scalar("A_g_loss", self.A_g_loss)
    ### define summary histogram
    #self.AD_predicts_real_hist = tf.summary.histogram('AD_predicts_real_hist', self.AD_predicts_real)
    #self.AD_predicts_fake_hist = tf.summary.histogram('AD_predicts_fake_hist', self.AD_predicts_fake)
    #self.ADC_predicts_real_hist = tf.summary.histogram('ADC_predicts_real_hist', self.ADC_predicts_real)
    #self.ADC_predicts_fake_hist = tf.summary.histogram('ADC_predicts_fake_hist', self.ADC_predicts_fake)
    ### define summary image
    self.real_A_summary_image = tf.summary.image('real_A', self.real_A)
    self.fake_A_summary_image = tf.summary.image('fake_A', self.fake_A)
    self.real_A_contour_summary_image = tf.summary.image('real_A_contour', self.real_A_contour)
    self.fake_A_contour_summary_image = tf.summary.image('fake_A_contour', self.fake_A_contour)

    self.d_loss_sum = tf.summary.merge([self.A_d_loss_sum
        #self.AD_predicts_real_hist,
        #self.AD_predicts_fake_hist,
        #self.ADC_predicts_real_hist,
        #self.ADC_predicts_fake_hist \
        ])
    self.g_loss_sum = tf.summary.merge([self.A_g_loss_sum, 
        self.A_loss_sum, 
        self.real_A_summary_image,
        self.fake_A_summary_image,
        self.real_A_contour_summary_image,
        self.fake_A_contour_summary_image])
    self.all_summaries = tf.summary.merge_all()

### define trainable variables
    t_vars = tf.trainable_variables()

    self.A_d_vars = [var for var in t_vars if 'A_d' in var.name]        
    self.A_g_vars = [var for var in t_vars if 'A_g' in var.name]
    self.d_vars = self.A_d_vars 
    self.g_vars = self.A_g_vars

    self.ga_opt_vars = self.A_g_vars
    
    self.gb_opt_vars = self.ga_opt_vars

    self.saver = tf.train.Saver(max_to_keep=2)
### Allocate GPU memory. because queue initializer will run them right after.
    self.sess.run(tf.global_variables_initializer()) 
    print('Memory allocated for all variables on GPU.')  


def run_singlegan_optim(self,start_time,step,niter=2):
    batch_A_images = self.rgb_flow.next()
    batch_B_images = self.hs_flow.next()

    _, Adfake,Adreal,Adcfake,Adcreal, Ad = \
                self.sess.run([self.d_optim, self.A_d_loss_fake, self.A_d_loss_real, self.A_d_C_loss_fake, self.A_d_C_loss_real, self.A_d_loss], \
                feed_dict = {self.real_A: self.Ad_queue.get_real(), self.fake_A: self.Ad_queue.get_real(),
                            self.real_A_contour: self.Adc_queue.get_fake(), self.fake_A_contour: self.Adc_queue.get_fake()})
    Ag=0
    for i in range(niter):
        _, Ag, Ad,fakea,realac,fakeac,summary_str = \
            self.sess.run([self.g_optim, self.A_g_loss, 
                self.A_d_loss, self.fake_A, 
                self.real_A_contour,self.fake_A_contour,self.all_summaries], 
                feed_dict={ self.real_A: batch_A_images, 
                            self.hsi_B: batch_B_images})

    self.Ad_queue.push(batch_A_images,fakea)
    self.Adc_queue.push(realac,fakeac)
    
    self.writer.add_summary(summary_str,step)

    print("time:%4.2f,A_d_loss:%04.2f,A_g_loss:%04.2f" \
                % (time() - start_time, Ad,Ag))


def singlegan_test(self, args):
    start_time = time()
    if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
        
    else:
        print(" [error] Load failed...")
        print(" aborted...")    
        return


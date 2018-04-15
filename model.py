############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

from __future__ import print_function
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import time as time_pkg
from time import time
from ops import *
from utils import *
from architectures import *
from model_fully_cyclegan import *
from model_began_cyclegan import *
from model_singlegan import *

class GAN(object):
    def __init__(self, sess, image_size=128, batch_size=1,fcn_filter_dim = 64,  
                 input_channels_A = 3, input_channels_B = 3, dataset='hsimg', 
                 checkpoint_dir=None, lambda_A = 200, lambda_B = 200, 
                 sample_dir=None, loss_metric = 'L1', 
                 flip = False, gtype='fcn',tag='',
                 *other_args,**other_kws):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training. [1]
            image_size: (optional) The resolution in pixels of the images. [128]
            fcn_filter_dim: (optional) Dimension of fcn filters in first conv layer. [64]
            input_channels_A: (RGB) Dimension of input image color of Network A. For grayscale input, set to 1. [3]
            input_channels_B: (HSI) automatically determined.
            A is assumed to be RGB
            B is assumed to be HSI, 
        """ 
        self.__dict__.update(locals())
        self.__dict__.update(other_kws)
        self.df_dim = fcn_filter_dim    
        self.is_grayscale_A = (input_channels_A == 1)
        self.is_grayscale_B = (input_channels_B == 1)
        # batch normalization : deals with poor initialization helps gradient flow
        
        #directory name for output and logs saving
        self.dir_name = \
        "%s-%s-batchSz_%s-imgSz_%s-fltrDim_%d-%s-lambdaAB_%s_%s-%s" % \
        (self.dataset, self.gtype, self.batch_size, self.image_size,self.fcn_filter_dim,\
        self.loss_metric, self.lambda_A, self.lambda_B, self.tag) 
        
        # prepare the data ignoring data dir 
        self.rgb_flow = get_img_generator(imgs_dir=os.path.join('datasets',self.dataset,'%s_%d'%(self.subdataset,self.image_size)),
                        img_size=image_size,
                        batch_size=batch_size)
        self.hs_flow = HsimgGenerator(filename=os.path.join('datasets',self.dataset,'dcmall/dc.tif'),
            img_size=image_size,
            batch_size=batch_size,
            flip=True,
            normalize=self.normalize)
        self.hsi_channels_B = self.hs_flow.channels
        
        if self.modeltype=='cyclegan':
            self.build_cyclegan_model()
            # cache queue for training Discriminator
            self.init_queue_for_cyclegan()
        elif self.modeltype=='fully_cyclegan':
            build_fully_cyclegan_model(self)
            self.init_queue_for_cyclegan()
        elif self.modeltype=='singlegan':
            build_singlegan_model(self)
            self.init_queue_for_singlegan()
        elif self.modeltype=='cyclebegan':
            build_BEGAN_model(self)
            self.init_queue_for_cyclegan()
        else:
            raise Exception('modeltype is not known %s'%self.modeltype)

        print('Queue for discriminator is well filled')
        
    def build_cyclegan_model(self):
        real_A = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_channels_A ],
                                        name='input_images_of_A_network')
        # B contains Hyperspectral Images
        hsi_B = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.hsi_channels_B ],
                                        name='input_images_of_B_network')
        # 3 channel HSI
        real_B = self.compressor(hsi_B,output_dim=self.input_channels_B)

    ### define graphs
        fake_B = translated_A = self.B_g_net(real_A, reuse = False)
        BD_predicts_fake = self.B_d_net(translated_A, reuse = False)
    
        fake_A = translated_B = self.A_g_net(real_B, reuse = False)
        AD_predicts_fake = self.A_d_net(translated_B, reuse = False)
    
    ### define structure loss (Not used in the final version)
        #hsi_gx2,hsi_gy2,hsi_gxy =hs_structures= structure_tensor(hsi_B,name='spectral',reuse=False)
        #cmp_gx2,cmp_gy2,cmp_gxy =rgb_strucures= structure_tensor(real_B,name='pseudo_rgb',reuse=False)
        #structure_loss = sum([tf.reduce_mean(tf.abs(hs-rgb)) \
        #                   for hs,rgb in zip(hs_structures,rgb_strucures)])
    ### variance loss
        variance_loss =  - tf.reduce_mean(tf.pow(real_B - tf.reduce_mean(real_B),2))
    ### define loss
        recover_A = self.A_g_net(translated_A, reuse = True)
        recover_B = self.B_g_net(translated_B, reuse = True)
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
        g_loss = A_g_loss + B_g_loss \
                + self.lambda_V*variance_loss
                #+ self.lambda_S*structure_loss 
    ### compressor loss
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
        real_B_summary_image = tf.summary.image('real_B', real_B)
        translated_A_summary_image = tf.summary.image('translated_A', translated_A)
        translated_B_summary_image = tf.summary.image('translated_B', translated_B)
        recover_A_summary_image = tf.summary.image('recover_A', recover_A)
        recover_B_summary_image = tf.summary.image('recover_B', recover_B)

        d_loss_sum = tf.summary.merge([A_d_loss_sum, B_d_loss_sum,
            BD_predicts_real_hist,
            BD_predicts_fake_hist,
            AD_predicts_real_hist,
            AD_predicts_fake_hist])
        g_loss_sum = tf.summary.merge([A_g_loss_sum, B_g_loss_sum, A_loss_sum, B_loss_sum, 
            real_A_summary_image,
            real_B_summary_image,
            translated_A_summary_image,
            translated_B_summary_image,
            recover_A_summary_image,
            recover_B_summary_image])
        all_summaries = tf.summary.merge_all()

        ## define trainable variables
        t_vars = tf.trainable_variables()

        A_d_vars = [var for var in t_vars if 'A_d' in var.name]
        B_d_vars = [var for var in t_vars if 'B_d' in var.name]
        
        A_g_vars = [var for var in t_vars if 'A_g' in var.name]
        B_g_vars = [var for var in t_vars if 'B_g' in var.name]
        
        c_vars = [var for var in t_vars if 'C_' in var.name]
        d_vars = A_d_vars + B_d_vars 
        g_vars = A_g_vars + B_g_vars + c_vars

        ga_opt_vars = A_g_vars + c_vars
        
        gb_opt_vars = B_g_vars

        saver = tf.train.Saver(max_to_keep=2)

        self.sess.run(tf.global_variables_initializer()) # pre-initialize here so that we can manipulate obj directly
        print('Memory allocated for all variables on GPU.')
        self.__dict__.update(locals())

    ################### Queue ###################
    def init_queue_for_cyclegan(self):
        self.Ad_queue = BatchQueue(self.d_queue_len,img_size=self.image_size,channels=self.real_A.get_shape()[-1])
        self.Bd_queue = BatchQueue(self.d_queue_len,img_size=self.image_size,channels=self.real_B.get_shape()[-1])
        reala,fakea,realb,fakeb = self.gen_init_batch_for_discriminator()
        self.Ad_queue.init_with_data(reala,fakea)
        self.Bd_queue.init_with_data(realb,fakeb)

    def init_queue_for_singlegan(self):
        self.Ad_queue = BatchQueue(self.d_queue_len,img_size=self.image_size)
        self.Adc_queue = BatchQueue(self.d_queue_len,img_size=self.image_size)
        reala,fakea,realac,fakeac = self.gen_init_batch_for_discriminator(two_queue=False)
        self.Ad_queue.init_with_data(reala,fakea)
        self.Adc_queue.init_with_data(realac,fakeac)

    def gen_init_batch_for_discriminator(self,two_queue=True):
        '''parameters
        use two_queue to init queue for cyclegan.
        two_queue=False to init only one queue for A
        '''
        real_A = []
        for i in range(self.d_queue_len):
            real_A.append(self.rgb_flow.next())
        real_A = np.concatenate(real_A,0) # matrix of  d_queue_len x height x width x channels
        hsi = self.hs_flow.next(self.d_queue_len)
        if two_queue:
            fake_B,fake_A,real_B = self.sess.run([self.fake_B,self.fake_A,self.real_B], 
                feed_dict={self.real_A:real_A,self.hsi_B:hsi})
            return real_A,fake_A,real_B,fake_B
        else:
            fake_A,real_A_contour,fake_A_contour = self.sess.run([self.translated_B,self.real_A_contour,self.fake_A_contour], 
                feed_dict={self.hsi_B:hsi,self.real_A:real_A})
            return real_A,fake_A,real_A_contour,fake_A_contour

    ################### Trainings ###################
    def train(self,args):
        # Record commands, time and details of this test run.
        self.record_experiment()
        if args.subphase=='compressor':
            print('creating optimizer for run_compressor_optim')
            self.Bd_optim = tf.train.AdamOptimizer(args.lr)\
                                .minimize(self.B_d_loss, var_list=self.B_d_vars)
            self.c_optim = tf.train.AdamOptimizer(args.lr)\
                                .minimize(self.C_loss, var_list=self.c_vars)  
        else:
            if 'd_optim' not in self.__dict__:
                print('initializing default d optimizer:adam')
                self.d_optim = tf.train.AdamOptimizer(args.lr)\
                                .minimize(self.d_loss, var_list=self.d_vars)
            if 'g_optim' not in self.__dict__:
                print('initializing default g optimizer:adam')
                self.g_optim = tf.train.AdamOptimizer(args.glr)\
                                .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run(session=self.sess)
        # draw computation graph to tensorboard
        self.writer = tf.summary.FileWriter("./logs/"+self.dir_name, self.sess.graph)

        start_time = time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" Load failed...neglected")
            print(" start training...") 
        rgb_epoch_size = self.rgb_flow.n
        hs_epoch_size = self.hs_flow.epoch_size
        epoch_size = min(rgb_epoch_size,hs_epoch_size)
        print('rgb epoch:',rgb_epoch_size,'hs epoch:',hs_epoch_size)

        print('[*] run optimizor...')
        for e_num in range(args.epoch):
            for i in range(epoch_size):
                mins_passed = (time() - start_time) / 60
                print("Epoch: [%2d] [%4d/%4d], Elapsed %.1f|%d" % (e_num,i,epoch_size,mins_passed,args.mins or 0) )
                self.optimize(start_time,
                    step=e_num*epoch_size+i+args.start_step,
                    niter=args.niter)

            if np.mod(e_num, 2) == 1:
                self.shotcut(self.rgb_flow.next(),self.hs_flow.next(),args.sample_dir, e_num, i, int(mins_passed))

            if np.mod(e_num, args.save_latest_freq) == 0 or (args.mins and mins_passed > args.mins):
                self.save(args.checkpoint_dir, e_num)

            if args.mins and ( mins_passed > args.mins ):
                return

    def optimize(self,*argv,**kw):
        if self.subphase == 'compressor':
            self.run_compressor_optim(*argv,**kw)
        elif self.modeltype in ['cyclegan','cyclebegan','fully_cyclegan']:
            self.run_cyclegan_optim(*argv,**kw)
        elif self.modeltype=='singlegan':
            run_singlegan_optim(self,*argv,**kw)
        else:
            pass

    def run_compressor_optim(self,start_time,step,niter=2):
        batch_A_images = self.rgb_flow.next()
        batch_B_images = self.hs_flow.next()
                                
        self.sess.run([self.Bd_optim], \
                    feed_dict = {self.real_B: self.Bd_queue.get_real(),
                                self.translated_A: self.Bd_queue.get_fake()})

        for i in range(niter-1):
            self.sess.run([self.c_optim], 
                feed_dict={ self.real_A: batch_A_images, \
                            self.hsi_B: batch_B_images})

        _, realb,fakeb, summary_str = \
            self.sess.run([self.c_optim,self.real_B,self.translated_A,\
            self.all_summaries], feed_dict={ self.real_A: batch_A_images, \
            self.hsi_B: batch_B_images})

        self.Bd_queue.push(realb,fakeb)
        
        self.writer.add_summary(summary_str,step)

    def run_cyclegan_optim(self,start_time,step,niter=2):
        batch_A_images = self.rgb_flow.next()
        batch_B_images = self.hs_flow.next()
        print('train D start')
        self.sess.run([self.d_optim], \
                    feed_dict = {self.real_A: self.Ad_queue.get_real(), self.real_B: self.Bd_queue.get_real(),
                                self.translated_A: self.Bd_queue.get_fake(), self.translated_B: self.Ad_queue.get_fake()})

        print('train G start')
        for i in range(niter-1):
            self.sess.run([self.g_optim], feed_dict={ self.real_A: batch_A_images, \
                self.hsi_B: batch_B_images})

        _, realb, fakea, fakeb, summary_str = \
            self.sess.run([self.g_optim, 
                self.real_B, 
                self.fake_A, 
                self.fake_B, 
                self.all_summaries], 
                feed_dict={ self.real_A: batch_A_images, \
                        self.hsi_B: batch_B_images}
                )
        self.Ad_queue.push(batch_A_images,fakea)
        self.Bd_queue.push(realb,fakeb)
        self.writer.add_summary(summary_str,step)
    
    ####################### Nets #######################
    def A_d_net(self, imgs, prefix='A_d', reuse = False):
        return self.discriminator(imgs, prefix = prefix, reuse = reuse)
    
    def B_d_net(self, imgs, prefix='B_d',reuse = False):
        return self.discriminator(imgs, prefix = prefix, reuse = reuse)
        
    def A_g_net(self, imgs, prefix='A_g', reuse=False):
        return self.g_net(imgs, prefix, reuse=reuse)

    def B_g_net(self, imgs, prefix='B_g',reuse=False):
        return self.g_net(imgs, prefix, reuse=reuse)

    def g_net(self,imgs,prefix,reuse=False):
        '''g nets are declared in architectures.py'''
        if self.gtype == 'cfc': # deprecated
            return cfc(imgs, prefix=prefix, reuse = reuse)
        elif self.gtype == 'fcn': # deprecated
            return fcn(imgs, prefix=prefix, reuse = reuse)
        elif self.gtype == 'pixnet':
            return pixnet(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'pixnet_residual':
            return pixnet_residual(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'pixnet_resnet':
            return pixnet_resnet(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'pixnet_mutichannel':
            return pixnet_mutichannel(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'squeeze_net':
            return squeeze_net(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'pixnet_firemodule':
            return pixnet_firemodule(imgs, prefix=prefix, reuse=reuse)
        elif self.gtype == 'pixnet_compressor':
            return pixnet_compressor(imgs, prefix=prefix, reuse=reuse)
        else:
            raise Exception('未知网络结构:%s' % self.gtype)

    def discriminator(self, image, prefix='A_d', reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(prefix,reuse=reuse):

            h0 = lrelu(conv2d(image, self.df_dim, name='h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name='h1_conv'), name = 'bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name='h2_conv'), name =  'bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, strides=1, name='h3_conv'), name =  'd_bn3'))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = conv2d(h3, 1, strides=1, name =prefix+'h4')
            return h4

    def compressor(self, image, output_dim=3, prefix='C_', reuse=False):
        if self.ctype=='simple':
            return simple(image=image,output_dim=output_dim,prefix=prefix,reuse=reuse)
        elif self.ctype=='resnet':
            return resnet_compress(image=image,output_dim=output_dim,prefix=prefix,reuse=reuse)
        elif self.ctype=='ccctanh':
            return ccctanh(image=image,output_dim=output_dim,prefix=prefix,reuse=reuse)


    ################## saveload ###################
    def save(self, checkpoint_dir, step):
        model_name = "DualNet.model"
        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir =  self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def record_experiment(self,logfilename='record.txt'):
        try:
            logfilepath = os.path.join('records',logfilename)
            if not os.path.exists(logfilepath):
                f = open(logfilepath,'w')
                f.write('科研实验开始::\n')
                f.close()

            with open(logfilepath,'a') as f:
                t = time_pkg.localtime()
                timestr = 'y:%d m:%d d:%d %d:%d:%d' %(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
                command = ' '.join(self.argv)
                template = \
                '''
-----------------
时间：%s
    命令：%s
    目录名：%s
    效果与结果备注：%s
    原因分析：
    示例：

                ''' % (timestr,command,self.dir_name,self.comment)
                f.write(template)
        except Exception as e:
            print('record_experiment:智能实验助理记录出错了。',e)
        else:
            print('record_experiment:智能实验助理已经帮您记录此次试验。')

    ################## shotcut ###################
    def shotcut(self,*argv,**kw):
        if self.modeltype=='singlegan':
            self.shotcut_singlegan(*argv,**kw)
        elif self.modeltype=='cyclegan':
            self.shotcut_cyclegan(*argv,**kw)
        else:
            pass

    def shotcut_cyclegan(self,imageAs,imageBs,sample_dir, epoch, idx, batch_idxs):
        sample_A_imgs,sample_B_imgs = imageAs,imageBs
        Ag, recover_A_value, translated_A_value = self.sess.run([self.A_loss, self.recover_A, self.translated_A], feed_dict={self.real_A: sample_A_imgs, self.hsi_B: sample_B_imgs})
        
        Bg, recover_B_value, translated_B_value = self.sess.run([self.B_loss, self.recover_B, self.translated_B], feed_dict={self.real_A: sample_A_imgs, self.hsi_B: sample_B_imgs})

        save_images(translated_A_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_A2B_{:02d}.png'.format(sample_dir,self.dir_name , epoch, idx, batch_idxs))
        save_images(recover_A_value, [self.batch_size,1],    './{}/{}/{:06d}_{:04d}_A2B2A_{:02d}_.png'.format(sample_dir,self.dir_name, epoch,  idx, batch_idxs))
        
        save_images(translated_B_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_B2A_{:02d}.png'.format(sample_dir,self.dir_name, epoch, idx,batch_idxs))
        save_images(recover_B_value, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_B2A2B_{:02d}.png'.format(sample_dir,self.dir_name, epoch, idx, batch_idxs))
        
        print("[Sample] A_loss: {:.8f}, B_loss: {:.8f}".format(Ag, Bg))

    def shotcut_singlegan(self,realA,hsi,sample_dir, epoch, idx, batch_idxs):
        sample_A_imgs,sample_B_imgs = realA,hsi
        Ag, fake_A, real_A_contour,fake_A_contour = self.sess.run([self.A_loss, self.fake_A, self.real_A_contour, self.fake_A_contour], 
            feed_dict={self.real_A: sample_A_imgs, self.hsi_B: sample_B_imgs})
        save_images(fake_A, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_B2A_{:02d}.png'.format(sample_dir,self.dir_name , epoch, idx, batch_idxs))        
        save_images(real_A_contour, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_AC_{:02d}.png'.format(sample_dir,self.dir_name, epoch, idx,batch_idxs))
        save_images(fake_A_contour, [self.batch_size,1], './{}/{}/{:06d}_{:04d}_BC_{:02d}.png'.format(sample_dir,self.dir_name, epoch, idx,batch_idxs))
        print("[Sample] A_loss: {:.8f}".format(Ag))

    def sample_shotcut(self, sample_dir, epoch, idx, batch_idxs):
        sample_A_imgs,sample_B_imgs = self.load_random_samples()
        self.shotcut(sample_A_imgs,sample_B_imgs,sample_dir, epoch, idx, batch_idxs)


    ################## TEST ###################
    def test(self,args):
        if self.modeltype=='singlegan':
            singlegan_test(self,args)
        elif self.modeltype in ['cyclegan','cyclebegan']:
            self.cyclegan_test(args)
        elif self.modeltype == 'fully_cyclegan':
            fully_cyclegan_test(self,args)
        else:
            pass

    def cyclegan_test(self,args):
        start_time = time()
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
        compressed_B,fake_A,fake_B,recover_A=self.sess.run([self.real_B,self.translated_B,
            self.translated_A,self.recover_A],
            feed_dict={self.real_A:rgb_puzzles,self.hsi_B:hs_puzzles})
        print('recovering')
        real2compressed = rgb_slicer.recover(fake_B,scale=True)
        hsi2compressed = hs_slicer.recover(compressed_B,scale=True)
        compressed2real = hs_slicer.recover(fake_A,scale=True)
        real2comp2real = rgb_slicer.recover(recover_A,scale=True)

        force_exist(os.path.join('test',self.dir_name))

        sm.imsave(os.path.join('test',self.dir_name,'real2compressed.png'),real2compressed)
        sm.imsave(os.path.join('test',self.dir_name,'hsi2compressed.png'),hsi2compressed)
        sm.imsave(os.path.join('test',self.dir_name,'compressed2real.png'),compressed2real)
        sm.imsave(os.path.join('test',self.dir_name,'real2comp2real.png'),real2comp2real)
        print('done')


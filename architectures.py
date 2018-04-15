############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

import numpy as np
import tensorflow as tf
from ops import *
slim = tf.contrib.slim
#adam
# ('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
# ('--d_lr', type=float, default=0.00008)
# ('--g_lr', type=float, default=0.00008)
# ('--lr_lower_boundary', type=float, default=0.00002)
# ('--beta1', type=float, default=0.5)
# ('--beta2', type=float, default=0.999)
# ('--gamma', type=float, default=0.5)
# ('--lambda_k', type=float, default=0.001)
# ('--use_gpu', type=str2bool, default=True)

def DiscriminatorAutoEncoder(x, input_channel=3, z_num=64, 
    repeat_num=None, hidden_num=32, 
    name='DiscAE'):
    _,height,w,c = x.get_shape().as_list()
    repeat_num = repeat_num or int(np.log2(height)) - 2
    with tf.variable_scope(name):
        # Encoder
        #x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = int(hidden_num * (idx*0 + 1))
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
            #x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = tf.reshape(x, [-1,8,8,hidden_num])
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            #x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu)
            if idx < repeat_num - 1:
                x = upscale(x, 2)

        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None)
    return out#, z

def D_AE(imgs,name='A_d',reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        return DiscriminatorAutoEncoder(imgs)

def upscale(x, scale):
    _, h, w, _ = x.get_shape().as_list()
    x = tf.image.resize_nearest_neighbor(x, (h*scale, w*scale))
    return x

def simple(image, output_dim=3, prefix='C_',activation=relu,reuse=False):
    _,h,w,c = image.get_shape().as_list()
    with tf.variable_scope(prefix,reuse=reuse):
        x = tf.layers.conv2d(image, filters=c//4, kernel_size=1)
        x = batch_norm(x, name='bn1')
        x = activation(x)
        x = tf.layers.conv2d(x,filters=output_dim, kernel_size=1)
        x = batch_norm(x, name='bn2')
        x = tf.tanh(x)
        #x = (x + 1) * 0.5 # scale from -1,1 to 0,1
        return x

def ccctanh(image, output_dim=3, prefix='C_',activation=tf.nn.elu,reuse=False):
    '''conv -> conv -> conv -tanh'''
    _,h,w,c = image.get_shape().as_list()
    with tf.variable_scope(prefix,reuse=reuse):
        x = tf.layers.conv2d(image, filters=c//2, kernel_size=1) # 45 channels
        x = activation(x)
        x = batch_norm(x, name='bn1')
        x = tf.layers.conv2d(image, filters=c//4, kernel_size=1) # 11
        x = activation(x)
        x = batch_norm(x, name='bn2')
        x = tf.layers.conv2d(x,filters=output_dim, kernel_size=1) # 3
        x = tf.tanh(x)
        return x

def simple_extender(image,output_dim=191, prefix='E_',activation=tf.nn.elu,reuse=False):
    _,h,w,c = image.get_shape().as_list()
    with tf.variable_scope(prefix,reuse=reuse):
        x = tf.layers.conv2d(image, filters=output_dim//16, kernel_size=1)
        x = activation(x)
        x = batch_norm(x, name='bn1')
        x = tf.layers.conv2d(image, filters=output_dim//4, kernel_size=1)
        x = activation(x)
        x = batch_norm(x, name='bn2')
        x = tf.layers.conv2d(x,filters=output_dim, kernel_size=1)
        x = tf.tanh(x)
        return x

def resnet_compress(image, output_dim=3, prefix='C_', reuse=False):
    _,h,w,c = image.get_shape().as_list()
    with tf.variable_scope(prefix,reuse=reuse):
        # resnet1
        x = tf.layers.conv2d(image, filters=c//16, kernel_size=1)
        dx = tf.layers.conv2d(image, filters=c//4, kernel_size=1)
        dx = relu(dx)
        dx = batch_norm(dx, name='bn1')
        dx = tf.layers.conv2d(dx, filters=c//16, kernel_size=1)
        res1 = tf.nn.elu(x+dx)
        res1 = batch_norm(res1)
        # resnet2
        x = tf.layers.conv2d(res1,filters=output_dim, kernel_size=1)
        dx = tf.layers.conv2d(res1,filters=c//32, kernel_size=1)
        dx = relu(dx)
        dx = batch_norm(dx, name='bn1')
        dx = tf.layers.conv2d(dx,filters=output_dim, kernel_size=1)
        res2 = tf.tanh(x+dx)
        return res2

def pixnet_compressor(image, output_dim=3, layers=8, prefix='PC_', reuse=False):
    _,h,w,c = image.get_shape().as_list()
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers):
            image = tf.layers.conv2d(image, 
                        filters=output_dim*(2*(layers-i)),
                        kernel_size=1,strides=1,padding='same')
            image = batch_norm(image)
            image = relu(image)
            image = residual(image,kernel_size=1,name='res_%d'%i)
        image = tf.layers.conv2d(image, filters=output_dim, kernel_size=1)
        image = tf.tanh(batch_norm(image))
        return image

def pixnet(image,layers=5,output_dim=3,prefix='P_', reuse=False):
    '''pretty good, but color'''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
            image = batch_norm(image,name='bn%d'%i)
            image = lrelu(image)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image,name='bn%d'%(layers-1))
        image = tf.tanh(image)
        return image

def pixnet_residual(image,layers=5,output_dim=3,prefix='PR_',reuse=False):
    '''x+d_pix+d_convx_restore
    tried,bad
    '''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = residual(x=image,kernel_size=1,name='e_res%d' %i,activation=tf.nn.elu)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image)
        image = tf.tanh(image)
        return image

def pixnet_resnet(image,layers=5,output_dim=3,prefix='PR_',reuse=False):
    '''x+d_pix+d_convx_restore
    tried,bad
    '''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = resnet(x=image,kernel_size=1,name='e_res%d' %i,activation=tf.nn.elu)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = tf.tanh(image)
        return image

def pixnet_elastic_residual(image,layers=5,output_dim=3,prefix='PR_',reuse=False):
    '''x+d_pix+d_convx_restore
    tried,bad
    '''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = elastic_residual(x=image,name='e_res%d' %i)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image)
        image = tf.tanh(image)
        return image

def pixnet_mutichannel(image,layers=5,output_dim=3,prefix='P_', reuse=False):
    '''fully pixel net, with extra channels'''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = tf.layers.conv2d(image,filters=output_dim*(2**(layers-i)),kernel_size=1)
            image = batch_norm(image,name='bn%d'%i)
            image = lrelu(image)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image,name='bn%d'%(layers-1))
        image = tf.tanh(image)
        return image

def squeeze_net(image,layers=5,output_dim=3,prefix='SQ_',reuse=False):
    '''x + conv(x)'''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = squeeze_net_module(x=image,output_dim=output_dim*(layers-i),name='sqz%d' %i)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image)
        image = tf.tanh(image)
        return image

def pixnet_firemodule(image,layers=5,output_dim=3,prefix='PR_',reuse=False):
    '''conv1by1(x) + conv3by3(x) lrelu'''
    with tf.variable_scope(prefix,reuse=reuse):
        for i in range(layers-1):
            image = fire_module(x=image,output_dim=output_dim*(2**(layers-i)),kernel_size=3,name='res%d' %i)
        image = tf.layers.conv2d(image,filters=output_dim,kernel_size=1)
        image = batch_norm(image)
        image = tf.tanh(image)
        return image
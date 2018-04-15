############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################


import numpy as np 
import tensorflow as tf

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.maximum(x, 0)
  
def batch_norm(x, train=True, name = "batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)

def conv2d(input_, output_dim, 
           kernel_size=5, strides=2, stddev=0.02,
           name="conv2d"):
    d_w = d_h = strides
    k_h = k_w = kernel_size
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def residual(x, kernel_size=5, stddev=0.02, name='res',activation=relu):
    b,h,w,c = x.get_shape().as_list()
    with tf.variable_scope(name):
        dx = tf.layers.conv2d(inputs=x,filters=c,kernel_size=kernel_size,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x2 = dx+x
        return activation(batch_norm(x2))

def resnet(x, kernel_size=5, stddev=0.02, name='res',activation=relu):
    b,h,w,c = x.get_shape().as_list()
    with tf.variable_scope(name):
        dx = tf.layers.conv2d(inputs=x,filters=c,kernel_size=kernel_size,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        dx = batch_norm(activation(dx))
        dx = tf.layers.conv2d(inputs=x,filters=c,kernel_size=kernel_size,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x2 = dx+x
        return batch_norm(activation(x2))

def elastic_residual(x, kernel_size=5, stddev=0.02, name='res',activation=relu):
    b,h,w,c = x.get_shape().as_list()
    with tf.variable_scope(name):
        d_pix = tf.layers.conv2d(inputs=x,
            filters=c,
            kernel_size=1,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        d_convx_down = tf.layers.conv2d(inputs=x,
            filters=c*2,
            kernel_size=kernel_size,
            strides=2,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        d_convx_down_activated = lrelu(batch_norm(d_convx_down))
        d_convx_restore = tf.layers.conv2d_transpose(inputs=d_convx_down_activated,
            filters=c,
            kernel_size=kernel_size,
            strides=2,padding='SAME',
            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
            bias_initializer=tf.constant_initializer(0,0))
        _,r_h,r_w,_ = d_convx_restore.get_shape().as_list()
        if (r_h != h) or (r_w != w):
            raise Exception('elastic_residual:tansposeConv2D:restore Failed:%d %d != %d %d' %(r_h,r_w,h,w))
        elas_x = x+d_pix+d_convx_restore
        return activation(batch_norm(elas_x))

def fire_module(x, output_dim, kernel_size=3, stddev=0.02, name='res',activation=lrelu):
    b,h,w,c = x.get_shape().as_list()
    with tf.variable_scope(name):
        dx = tf.layers.conv2d(inputs=x,filters=output_dim,kernel_size=kernel_size,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        pix_x = tf.layers.conv2d(inputs=x,filters=output_dim,kernel_size=1,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        x2 = dx+pix_x
        return activation(batch_norm(x2))

def squeeze_net_module(x, output_dim, width=3, stddev=0.02, name='res'):
    b,h,w,c = x.get_shape().as_list()
    with tf.variable_scope(name):
        dxs = [ tf.layers.conv2d(inputs=x,filters=output_dim,kernel_size=1,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
                for i in range(width)]

        d3x = tf.layers.conv2d(inputs=x,filters=output_dim,kernel_size=3,
            strides=1,padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        merged = tf.add_n(dxs) + d3x
        return relu(batch_norm(merged))

def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def sum_cross_entropy(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def replicate_3d_kernel(kernel_mat,in_channels=3,name='replicate_3d_kernel'):
    '''explaination:
    use same 2d kernel kernel_mat to process all channels(in_channels),
    and output only 1 channel.'''
    with tf.variable_scope(name):
        mat = np.array(kernel_mat)
        kernel_size = mat.shape[0]
        kernel = np.zeros((kernel_size,kernel_size,in_channels,in_channels))
        for i in range(in_channels):
            kernel[:,:,i,i] = mat
        return tf.get_variable(name='replicate_3d_kernel',
            initializer=tf.constant_initializer(kernel),
            shape=kernel.shape,
            trainable=False,dtype=tf.float32)

def group_kernel(kernel_mat,name='group_kernel'):
    '''group kernel are n by n square matrix. not tensor
    this will automatically reshape the matrix into nxnx1x1 tensor
    '''
    with tf.variable_scope(name):
        knp = np.array(kernel_mat)
        knp_1_1 = knp.reshape(knp.shape+(1,1)) # (height,width,inchannels=1,outchannels=1)
        return tf.get_variable(name='constant_kernel',shape=knp_1_1.shape,
            initializer=tf.constant_initializer(knp_1_1),
            dtype=tf.float32,trainable=False)

def contour(img,reuse=False,name='rgb',return_all=False):
    b,h,w,c = img.get_shape().as_list()
    with tf.variable_scope('contour_'+name,reuse=reuse):
        x_kernel = replicate_3d_kernel([[-1,0,1],
            [-1,0,1],
            [-1,0,1]],
            in_channels=c,
            name='x_kernel_np')
        y_kernel = tf.transpose(x_kernel,[1,0,2,3])
        x_conv = tf.nn.conv2d(input=img,filter=x_kernel, \
                                                        strides=(1,1,1,1),padding='SAME',name='x_conv')
        y_conv = tf.nn.conv2d(input=img,filter=y_kernel, \
                                                        strides=(1,1,1,1),padding='SAME',name='y_conv')
        x_conv_norm = x_conv/c
        y_conv_norm = y_conv/c
        x_conv2 = tf.square(x_conv_norm)
        y_conv2 = tf.square(y_conv_norm)

        grad = tf.pow(x_conv2 + y_conv2 + 1e-6,0.5)
        if return_all:
            return grad,x_conv_norm,y_conv_norm
        else:
            return grad

def structure_tensor(img,name='rgb',reuse=False):
    with tf.variable_scope('strucutre_tensor',reuse=reuse):
        dimg,xgrad_nchannel,ygrad_nchannel = contour(img,name=name,return_all=True,reuse=reuse)
        xgrad = tf.reduce_mean(xgrad_nchannel,axis=3) #
        ygrad = tf.reduce_mean(ygrad_nchannel,axis=3) # reduce to 1 channel
        gx2 = tf.square(xgrad)
        gy2 = tf.square(ygrad)
        gxy = xgrad*ygrad
        return gx2,gy2,gxy

def contour_demo(filename,reuse=False,name='rgb'):
    import scipy.misc as sm
    if isinstance(filename,str):
        img = sm.imread(filename)
    else:
        img = filename
    if len(img.shape) ==3:
        img = img.reshape((1,)+img.shape) # add batch axis
    b,h,w,c = img.shape
    with tf.variable_scope(name,reuse=reuse):
        in_tensor = tf.placeholder(tf.float32, shape=img.shape)
        dimg,xgrad_nchannel,ygrad_nchannel = contour(in_tensor,return_all=True,name='',reuse=reuse)
        xgrad = tf.reduce_mean(xgrad_nchannel,axis=3)
        ygrad = tf.reduce_mean(ygrad_nchannel,axis=3)
        gx2 = tf.square(xgrad)
        gy2 = tf.square(ygrad)
        gxy = xgrad*ygrad
        return img,in_tensor,gx2,gy2,gxy


if __name__ == '__main__':
    import tifffile
    img = tifffile.imread('./datasets/hsimg/dcmall/dc.tif')
    img,in_tensor,gx2,gy2,gxy = contour_demo('datasets/hsimg/registered_DC_large.bmp')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run([gx2,gy2,gxy],feed_dict={in_tensor:img})
    np.save('res',res)


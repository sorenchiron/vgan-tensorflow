############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################


from __future__ import print_function
import argparse
import os
import scipy.misc
import numpy as np

from model import GAN
from utils import force_exist
from sys import argv
import tensorflow as tf

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
#parser.add_argument('--network_type', dest='network_type', default='fcn_4', help='fcn_1,fcn_2,fcn_4,fcn_8, fcn_16, fcn_32, fcn_64, fcn_128')
parser.add_argument('--image-size', dest='image_size', type=int, default=128, help='size of input images (applicable to both A images and B images)')
parser.add_argument('--fcn-filter-dim', dest='fcn_filter_dim', type=int, default=64, help='# of fcn filters in first conv layer')
parser.add_argument('--gtype', dest='gtype', default='pixnet', help='network type of Generative model: fcn|cfc|pixnet|pixnet_residual|pixnet_mutichannel|squeeze_net|pixnet_firemodule')
parser.add_argument('--ctype', dest='ctype', default='simple', help='network type of Compressor model')
parser.add_argument('--modeltype', dest='modeltype', default='cyclegan', help='model type: cyclegan|singlegan')
"""Arguments related to run mode"""
parser.add_argument('--phase', dest='phase', default='modern', help='modern, train, test')
parser.add_argument('--subphase', dest='subphase', default='', help='compressor. to reverse train the compressor')

"""Arguments related to training"""
parser.add_argument('--loss-metric', dest='loss_metric', default='L1', help='L1, or L2')
parser.add_argument('--niter', dest='niter', type=int, default=2, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0005, help='D initial learning rate for adam')#0.0002
parser.add_argument('--glr', dest='glr', type=float, default=0.0009, help='G initial learning rate for adam')#0.0002
parser.add_argument('--galr', dest='galr', type=float, default=0.0009, help='G initial learning rate for adam')#0.0002
parser.add_argument('--gblr', dest='gblr', type=float, default=0.0009, help='G initial learning rate for adam')#0.0002
parser.add_argument('--clr', dest='clr', type=float, default=0.01, help='initial learning rate for adam')#0.0002
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', action='store_true', default=True, help='if flip the images for data argumentation')
parser.add_argument('--dataset', dest='dataset', default='hsimg', help='name of the dataset')
parser.add_argument('--subdataset', dest='subdataset', default='dcreal', help='Real style subdataset of hsimg(dcreal_|googlemaps_)')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--mins', dest='mins', type=int, default=None, help='# maximum traning time in minutes')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--start-step', dest='start_step', type=int, default=0, help='# starting step of optimization')
parser.add_argument('--d-queue-len', dest='d_queue_len', type=int, default=50, help='length of queue cached for batch-training Discriminator')
parser.add_argument('--lambda-A', dest='lambda_A', type=float, default=50.0, help='# weights of A recovery loss')
parser.add_argument('--lambda-B', dest='lambda_B', type=float, default=50.0, help='# weights of B recovery loss')
parser.add_argument('--lambda-C', dest='lambda_C', type=float, default=1.0, help='# weights of C Contour loss')
parser.add_argument('--lambda-S', dest='lambda_S', type=float, default=1e-6, help='# weights of structure loss')
parser.add_argument('--lambda-V', dest='lambda_V', type=float, default=1e-6, help='# weights of variance loss')
#parser.add_argument('--clamp', dest='clamp', type=float, default=0.01, help='#n_critic')

"""Arguments related to Data processing"""
parser.add_argument('--normalize', dest='normalize', action='store_true', default=False, help='normalize hsimg data')

"""Arguments related to monitoring and outputs"""
parser.add_argument('--save-epoch-freq', dest='save_epoch_freq', type=int, default=1, help='save a model every save_epoch_freq epochs')
parser.add_argument('--save-latest-freq', dest='save_latest_freq', type=int, default=1000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample-dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test-dir', dest='test_dir', default='./test', help='test sample are saved here')

"""Arguments related to Saving and restoring"""
parser.add_argument('--tag', dest='tag', default='', help='tag string append to save log path')
parser.add_argument('--comment', dest='comment', default='', help='comment string append to exp record')

"""Arguments unused"""
parser.add_argument('--input-channels-A', dest='input_channels_A', type=int, default=3, help='# of input image channels')
parser.add_argument('--input-channels-B', dest='input_channels_B', type=int, default=3, help='# of output image channels')


def main():
    args = parser.parse_args()
    force_exist(args.checkpoint_dir)
    force_exist(args.sample_dir)
    force_exist(args.test_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = GAN(sess,argv=argv,**args.__dict__)
        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        else:
            print('pass, exited')

if __name__ == '__main__':
    main()

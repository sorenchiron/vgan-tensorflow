############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
############################################################################

from main import parser
from model import *
import argparse
import scipy.misc as sm
# args of main to build GAN in model
args = parser.parse_args('--phase train --modeltype cyclegan --gtype pixnet_resnet --epoch 2 --image-size 128 --niter 2 --glr 0.0001 --lr 0.00001 --lambda-A 50 --lambda-B 50 --d-queue-len 100 --tag res_google'.split())
# args of this py script
this_parser = argparse.ArgumentParser(description="计算图片真实度")
this_parser.add_argument('--input',type=str,dest='input',nargs='+',help='path of input image')
this_parser.add_argument('--normalize',dest='normalize',action='store_true',help='scale the images into 0~1')
this_py_args = this_parser.parse_args()
config = tf.ConfigProto()
config.gpu_options.allow_growth =True


def main():
    sess = tf.Session(config=config)
    g = GAN(sess,argv=args,**args.__dict__)
    csv=[]
    for fname in this_py_args.input:
        img = sm.imread(fname)
        img = sm.imresize(img,[1280,306])
        puzzler = BigImagePuzzler(img=img[:,:,:3])
            # restrict the channels to 3
        batch = np.stack(puzzler.slice())
        if this_py_args.normalize:
            batch = batch / 255.0
        #print('inferring',fname)
        #print('shape of current slicing:',batch.shape)
        #res = sess.run(g.AD_predicts_real, feed_dict={g.real_A:batch})
        res = sess.run(g.A_d_loss_real, feed_dict={g.real_A:batch})
        loss = np.mean(res)
        print('ouput for',fname,batch.shape,'is',loss)
        csv.append(fname+','+str(loss))
    print("csv===========================")
    for line in csv:
        print(line)


if __name__ == '__main__':
    main()






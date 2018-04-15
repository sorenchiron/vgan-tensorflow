############################################################################
# VGAN: Spectral Image Visualization Using Generative Adversarial Networks
# LICENSE: MIT
# Author: CHEN SiYu
# DATE: 2017-2018
# Some code is from https://github.com/Newmu/dcgan_code
############################################################################

from __future__ import division,print_function
import math
import json
import random
import pprint
import scipy.misc
import scipy.misc as sm
import tifffile as tiff
import scipy.io as sio
import numpy as np
import os
from glob import glob
from time import gmtime, strftime
from tqdm import tqdm
import keras
from keras.preprocessing import image  

from math import log,floor


def save_images(images, size, image_path):
    dir = os.path.dirname(image_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True)#.astype(np.float)
    else:
        return scipy.misc.imread(path)#.astype(np.float)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) < 4:
        img = np.zeros((h * size[0], w * size[1], 1))
        images = np.expand_dims(images, axis = 3)
    else:
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    if images.shape[3] ==1: # turn grey image into pseudo-rgb
        return np.concatenate([img,img,img],axis=2)
    else:
        return img.astype(np.uint8) # keep rgb as it is

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return ((images+1.)*127.5)#/2.

################## saving tools ###################

def force_exist(dirname):
    if dirname == '' or dirname == '.':
        return True
    top = os.path.dirname(dirname)
    force_exist(top)
    if not os.path.exists(dirname):
        print('creating',dirname)
        os.makedirs(dirname)
        return False
    else:
        return True

def save_batch_imgs(imgs,path,pattern='%d.png'):
    '''imgs: 4-D array  batch x height x width x channels'''
    force_exist(path)
    for i,img in tqdm(enumerate(imgs)):
        sm.imsave(os.path.join(path,pattern%i),img)
    return True


############# Data augmentation generator ############

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def scale_tanh_2_floatimg(x):
    return (x+1)*0.5

def get_img_generator(imgs_dir='datasets/hsimg/dcreal_128',
    img_size=128,
    batch_size=1,
    channels=3):
    '''
    真实图像可以增强，高光谱图像难以增强，这个函数只给RGB用
    '''
    generator = image.ImageDataGenerator(
        preprocessing_function=preprocess_input, # force into range 0-1
        data_format='channels_last',
        rotation_range=180.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')

    flow_from_directory_params = {'target_size': (img_size, img_size),
                                  'color_mode': 'grayscale' if channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}

    flow = generator.flow_from_directory(
        directory=imgs_dir,
        **flow_from_directory_params)

    return flow

class HsimgGenerator():
    def __init__(self,filename='datasets/hsimg/dcmall/dc.tif',img_size=256,batch_size=1,flip=True,normalize=False):
        '''filenames: file pattern, or 1 filename, or list[string]
        self.imgs is list of np.array with  heightxwidthxchannels,
        image size must be exactly same'''
        self.__dict__.update(locals())
        self.imgs = self.read_spectral_images(filename)
        self.height,self.width,self.channels = self.imgs[0].shape
        self.batch_size = batch_size
        self.img_size = img_size
        self.flip = flip

        self.img_num = len(self.imgs)
        self.height_range = self.height - img_size
        self.width_range = self.width - img_size
        self.epoch_size = self.height_range * self.width_range * self.img_num
        print('HS img read:height %d width %d channels %d patches %d,nums %d' %(
            self.height,self.width,self.channels,self.epoch_size,self.img_num) )
        self.normalize = normalize
        if normalize:
            self.normalize_image()

    def normalize_image(self,mode=None):
        '''-1,1 image can be properly dealt by scipy misc
        if mode == rgb, then cmax,cmin will be set to 255,0'''
        delta = 1e-13
        #cmaxs = self.img.max(axis=(0,1)) + delta
        #cmins = self.img.min(axis=(0,1)) + delta
        imgs = []
        for img in self.imgs:
            cmaxs = img.max(axis=(0,1))+delta
            cmins = img.min(axis=(0,1))+delta
            #cavgs = self.img.sum(axis=(0,1)) / (self.height*self.width)
            #cvars = img.var(axis=(0,1)) + delta
            if mode == 'rgb':
                cmaxs = np.ones_like(cmaxs)*255
                cmins = np.zeros_like(cmins)
            tmp_img = (-0.5 + (img - cmins) / (cmaxs - cmins + delta))*2 # -1,1
            imgs.append(tmp_img)
        self.imgs = imgs
        return imgs

    def one_patch(self):
        ri = np.random.random()
        rh = np.random.uniform(low=0,high=1,size=1)
        rw = np.random.uniform(low=0,high=1,size=1)
        i = int((ri-1e-4) // (1/self.img_num))
        h = int(rh * self.height_range)
        w = int(rw * self.width_range)
        if (h%self.img_size) or (w%self.img_size):# patch in test
            return self.one_patch() # dice again
        else:
            patch = self.imgs[i][h:h+self.img_size,w:w+self.img_size,:]
            if self.flip:
                hf = np.random.rand()>0.5
                wf = np.random.rand()>0.5
                if hf:
                    patch = np.flip(patch,axis=1)
                if wf:
                    patch = np.flip(patch,axis=0)
            return patch.reshape((1,self.img_size,self.img_size,self.channels))

    def next(self, size=None):
        '''try extract only the first 3 channels from hsimg,
        just to test the integrity'''
        batch = np.concatenate([ self.one_patch() for i in range(size or self.batch_size)],axis=0)
        return batch

    def read_spectral_images(self,fnames):
        if isinstance(fnames,list) or isinstance(fnames,tuple):
            pass
        elif '*' in fnames: # fnames is a pattern "asdf/fe/sf/*.png"
            fnames = glob(fnames)
        else: # just one filename
            fnames = [fnames,]
        imgs=[]
        for fname in tqdm(fnames):
            suffix = os.path.basename(fname).split('.')[-1].lower()
            if suffix in ['tif','tiff']:
                img = tiff.imread(fname) # (191, 1280, 307)
                img = img.transpose([1,2,0]) # (1280, 307, 191)
                imgs.append(img)
            elif suffix in ['mat',]:
                imgs.append(sio.loadmat(fname)['DataCube']) #(960, 1280, 18)
            else:
                imgs.append(sm.imread(fname))
        return imgs

class BigImagePuzzler(HsimgGenerator):
    '''
    dense: cut crops one pixel by 
    '''
    def __init__(self,img,crop_size=128,dense=False,limit=None,normalize=False):
        '''parameters
        img: 3D np array or string (filename)
        dense: crop puzzles pixel by pixel
        '''
        self.img_name=None
        if isinstance(img,str):
            self.img_name=os.path.basename(img).split('.')[0]
            img = sm.imread(img)
        if len(img.shape)==4:
            print('警告：图片尺寸包含批处理维')
        imgs = [img]

        h,w,c = img.shape
        height,width = h,w
        h_normal_steps = h//crop_size
        w_normal_steps = w//crop_size

        if w_normal_steps==0:
            raise Exception('宽度%d不足以创建长宽为%d的小块'%(w,crop_size))
        elif h_normal_steps==0:
            raise Exception('高度%d不足以创建长宽为%d的小块'%(h,crop_size))
        
        self.__dict__.update(locals())
        if not dense:
            self.traverse = [i for i in self.traverse_image()]
        else:
            self.traverse = [i for i in self.dense_traverse()]
        self.batch_size = len(self.traverse)
        if limit:
            self.limit_batch(num=limit)
        if normalize:
            self.normalize_image(normalize)
            self.img = self.imgs[0]

    def traverse_image(self):
        crop_size=self.crop_size
        for h in range(self.h_normal_steps+1):
            for w in range(self.w_normal_steps+1):
                hstart = crop_size * h
                hend =  hstart + crop_size

                wstart = crop_size * w
                wend = wstart + crop_size 

                if h == self.h_normal_steps: # border case
                    hstart = self.h - crop_size
                    hend = self.h
                if w == self.w_normal_steps:
                    wstart = self.w - crop_size
                    wend = self.w
                yield (hstart,hend,wstart,wend)

    def slice(self):
        batch = np.zeros(shape=(self.batch_size,self.crop_size,self.crop_size,self.c))
        for i in range(self.batch_size):
            hs,he,ws,we = self.traverse[i]
            batch[i,:,:,:] = self.img[hs:he,ws:we,:]
        return batch

    def recover(self,imgs,scale=False):
        """parameters:
        imgs: 4-D np array
        scale: return (img-1)/2
        """
        b,h,w,c = imgs.shape
        img = np.zeros(shape=(self.h,self.w,c))

        if b != self.batch_size:
            print('小图片数量%d与所%d需不一致' %(b,self.batch_size))
            return None
        if h!=w or w!=self.crop_size:
            print('输入小图片长宽%d %d不对%d' %(h,w,self.crop_size))
            return None
        if c!=self.c:
            print('警告：输入通道数%d不对%d' %(c,self.c))

        for i in range(self.batch_size):
            hs,he,ws,we = self.traverse[i]
            img[hs:he,ws:we,:] = imgs[i,:,:,:]
        if scale:
            img = scale_tanh_2_floatimg(img)
        return img
        
    def dense_traverse(self):
        '''crop puzzles pixel by pixel'''
        crop_size = self.crop_size
        for h in range(self.h-crop_size):
            for w in range(self.w-crop_size):
                yield (h,h+crop_size,w,w+crop_size)

    def save_slices(self,slices,target_dir,prefix='img',start=0):
        '''start: starting number id for counting, only affects filenames'''
        prefix=prefix or self.img_name
        force_exist(target_dir)
        for i in tqdm(range(self.batch_size)):
            img = slices[i,:,:,:]
            sm.imsave(os.path.join(target_dir,'%s_%d.png'%(prefix,int(i+start) )),img)

    def limit_batch(self,num=2000):
        '''
        limit the number of cropped puzzles in case they are too many.
        '''
        np.random.shuffle(self.traverse)
        self.traverse = self.traverse[:num]
        self.batch_size = num
        return self.traverse

def batch_cut(dirpath='.',gather=None,crop_size=128,suffix='png',dense=True,limit=1000):
    '''gather: dirname to gather all patches, None if you want to keep each in separated folder'''
    from glob import glob
    filenames = glob(os.path.join(dirpath,"*."+suffix))
    if gather:
        gather = os.path.join(dirpath,gather)
    print(filenames)
    for filename in filenames:
        slicer = BigImagePuzzler(filename,crop_size=crop_size,dense=dense,limit=limit)
        img_name = slicer.img_name
        outdir = gather or os.path.join(dirpath,img_name)
        force_exist(outdir)
        slicer.save_slices(slicer.slice(),outdir,prefix=None)
    print('done')

#batch_cut('datasets/googlemap',gather='out',limit=500)

img_width = img_height = 256
channels = 3
depth = 64
conv_num = 5
batch_size = 1

input_shape = (img_height, img_width, channels)


#################################################
# functions
#################################################

def get_original_img_generator(A_class_dir='data/hsimg/train/A',
    B_class_dir='data/hsimg/train/B',
    img_height=img_height,
    img_width=img_width,
    batch_size=batch_size):
    generator = image.ImageDataGenerator(
        preprocessing_function=applications.xception.preprocess_input,
        data_format='channels_last')

    flow_from_directory_params = {'target_size': (img_height, img_width),
                                  'color_mode': 'grayscale' if channels == 1 else 'rgb',
                                  'class_mode': None,
                                  'batch_size': batch_size}
    gen_real = generator.flow_from_directory(
        directory=A_class_dir,
        **flow_from_directory_params)

    gen_hsi = generator.flow_from_directory(
        directory=B_class_dir,
        **flow_from_directory_params)
    return gen_real,gen_hsi

def get_image_batch(image_generator):
    '''
    output: (1, height, width) values from -1 to 1
    this generator automatically scale the rgb value from 255 to  -1~1
    so we don't need to scale it manually
    '''
    img_height, img_width, channels = image_generator.image_shape
    img_batch = image_generator.next()
    batch_size = image_generator.batch_size
    # keras generators may generate an incomplete batch for the last batch in an epoch of data
    if len(img_batch) != batch_size:
        img_batch = image_generator.next()
    assert img_batch.shape == (batch_size, img_height, img_width, channels)
    return img_batch


#################################################
# classes
#################################################
class BatchQueue(object):
    def __init__(self, max_length=50, y_max=1.5, y_min=0.8, 
        img_size=128,
        channels=channels):
        '''
            if max length is 50, then there are at most 25 real samples and 25 fake
            sample in the queue.
        '''
        self.real=[]
        self.fake=[]
        self.max_length=max_length//2
        self.real_generator=None
        self.y_max = y_max
        self.y_min = y_min
        self.width = img_size
        self.height = img_size
        self.channels = channels
        self.input_shape = (self.height, self.width, self.channels)
        self.real_generator=None
        self.base_generator=None
        self.faking_model=None

    def init_with_generator(self,real_generator, base_generator, faking_model):
        '''
        real will be marked as 1, 
        fake samples generated by base_generator will be marked as 0
        '''
        self.real_generator=real_generator
        self.base_generator=base_generator
        self.faking_model=faking_model
        for i in range(self.max_length):
            real = get_image_batch(real_generator)
            fake = faking_model.predict(get_image_batch(base_generator))
            self.push_real(real)
            self.push_fake(fake)
        return self

    def init_with_data(self,real,fake):
        self.push_real(real)
        self.push_fake(fake)

    def reinit(self):
        '''
        init the queue again after parameters are loaded by restore()
        '''
        return self.init(self.real_generator,self.base_generator,self.faking_model)

    def push(self,real,fake):
        self.push_real(real)
        self.push_fake(fake)

    def push_real(self,realimg):
        return self._push(self.real,realimg)

    def push_fake(self,fakeimg):
        return self._push(self.fake,fakeimg)    

    def _push(self, queue, imgs):
        shape = imgs.shape
        # batch of images is pushing in
        batch_size = shape[0]
        if shape[1:] != self.input_shape:
            print('BatchQueue[Warning]:input',shape,'inconsistent with',self.input_shape)
            #return False
        for i in range(batch_size):
            self.__push(queue,imgs[i:i+1,:,:,:])
        return True

    def __push(self, queue, img):
        if len(queue)< self.max_length:
            queue.insert(0,img)
        else:
            queue.pop()
            self._push(queue,img)
 
    def get(self,queue):
        '''
        input: List[np.vector]
        make an np.ndarray with size of: all x columns
        '''
        l=len(queue)
        dat=np.concatenate(queue,0)
        return dat,l

    def get_real(self):
        r,_ = self.get(self.real)
        return r
    def get_fake(self):
        r,_ = self.get(self.fake)
        return r

    def get_batch(self):
        '''
        get a batch with batch size of self.max_length*2
        '''
        realbatch, realbatchsize = self.get(self.real)
        fakebatch, fakebatchsize = self.get(self.fake)
        x=np.concatenate([realbatch,fakebatch],0)
        ones=np.random.uniform(low=self.y_min,high=self.y_max,size=realbatchsize)
        zeros=np.random.uniform(low=-self.y_max,high=-self.y_min,size=fakebatchsize)
        y=np.concatenate([ones,zeros],0)
        return x,y

    def get_real_fake(self):
        r,_ = self.get(self.real)
        f,_ = self.get(self.fake)
        return r,f

    def new_sample(self):
        '''
        returns one img or imgs according to generator's batch size configuration
        '''
        imgs = get_image_batch(self.real_generator) # ranges -1,1
        return imgs, np.random.uniform(low=self.y_min,high=self.y_max,size=imgs.shape[0])


def calculate_img_grad(img,sess,reuse=False,tag='3channels'):
    from ops import contour_demo
    import tensorflow as tf
    img = tf.placeholder(tf.float32, (None,)+pic.shape)
    inimg,outimg = contour_demo_rgb(contour_demo_rgb)
    if len(img.shape) ==3:
        img = img.reshape((1,)+img.shape) # add batch axis
    a=contour(img,return_all=True,name=tag,reuse=reuse)

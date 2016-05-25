# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:52:08 2016

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('/home/ztq/GitWorkSpace/caffePythonTest/')
sys.path.append('/home/ztq/caffe/python/')
import caffe
import CaffePythonTools as mytools
caffe.set_mode_gpu()
'''
net = caffe.Net('/home/ztq/caffe/examples/myfile/train_val_two.prototxt',\
                       '/home/ztq/caffe/examples/myfile/caffenet_two_iter_300.caffemodel',caffe.TEST)

net = caffe.Net('/home/ztq/caffe/examples/myfile/cifar10_quick_train_test.prototxt',\
                       '/home/ztq/caffe/examples/myfile/cifar10_iter_403.caffemodel',caffe.TEST)
'''
net = caffe.Net('/home/ztq/caffe/examples/myfile/cifar10_quick.prototxt',\
                       '/home/ztq/caffe/examples/myfile/cifar10_iter_403.caffemodel',caffe.TEST)                     
#img_src = caffe.io.load_image('/home/ztq/caffe/data/re/test/300.jpg')
#img_src = cv2.imread('/home/ztq/caffe/data/re/test/312.jpg')
img_src = plt.imread('/home/ztq/caffe/data/re/test/312.jpg')
#must change opencv-style BGR to caffe-style RGB

#img_src[:,:,0],img_src[:,:,2] = img_src[:,:,2].copy(),img_src[:,:,0].copy()
plt.imshow(img_src)
#im = np.zeros((256,256,3),dtype = np.uint8)
#cv2.resize(img_src,(256,256),im)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})#resize图像
transformer.set_transpose('data', (2,0,1))#交换array的每个维度，在这里将(256,256,3)换为(3,256,256)
transformer.set_mean('data', np.load('/home/ztq/caffe/examples/myfile/mean.npy').mean(1).mean(1)) # 减去均值
transformer.set_raw_scale('data', 1)  #所有数据乘以scale
transformer.set_channel_swap('data', (2,1,0))#交换图像的各个通道，在这里将RGB换为BGR，只对blob中的第二个位置，即(1,3,20,20)中的3中的通道互相交换
net.blobs['data'].data[...] = transformer.preprocess('data',img_src)
data = net.blobs['data'].data[0]
#plt.imshow(data.transpose(1,2,0))

print ''
print net.forward()
print ''

print [(k,v.data.shape) for k,v in net.blobs.items()]#data
print ' '
print [(k,v[0].data.shape) for k,v in net.params.items()]#weights                     
print [(k,v[1].data.shape) for k,v in net.params.items()]#biases

img_id =0
#use .copy() to protect the data being modified!!!
mytools.show_data(net.blobs['data'].data[img_id].copy(),'data image')
mytools.show_data(net.blobs['conv1'].data[img_id].copy(),'conv1 image')
mytools.show_data(net.blobs['pool1'].data[img_id].copy(),'pool1 image')
mytools.show_data(net.blobs['conv2'].data[img_id].copy(),'conv2 image')
mytools.show_data(net.blobs['pool2'].data[img_id].copy(),'pool2 image')
mytools.show_data(net.blobs['conv3'].data[img_id].copy(),'conv3 image')
mytools.show_data(net.blobs['pool3'].data[img_id].copy(),'pool3 image')

plt.figure()
plt.plot(net.blobs['prob'].data[img_id])
'''
mytools.show_feature(net.params['conv1'][0].data.reshape(32*3,5,5).copy(),'conv1 weights(filter)')
mytools.show_feature(net.params['conv2'][0].data.reshape(32*32,5,5).copy(),'conv2 weights(filter)')
mytools.show_feature(net.params['conv3'][0].data.reshape(64*32,5,5)[:1024].copy(),'conv3 weights(filter)')
'''

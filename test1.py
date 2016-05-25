# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:23:11 2016

@author: root
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ztq/GitWorkSpace/caffePythonTest/')
sys.path.append('/home/ztq/caffe/python/')
import caffe
import CaffePythonTools as mytools
caffe.set_mode_gpu()
name_list = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E',\
                'F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V',\
                'W','X','Y','Z','zh_cuan','zh_e','zh_gan','zh_gan1','zh_gui',\
                'zh_gui1','zh_hei','zh_hu','zh_ji','zh_jin','zh_jing','zh_jl',\
                'zh_liao','zh_lu','zh_meng','zh_min','zh_ning','zh_qing',\
                'zh_qiong','zh_shan','zh_su','zh_sx','zh_wan','zh_xiang',\
                'zh_xin','zh_yu','zh_yu1','zh_yue','zh_yun','zh_zang','zh_zhe']

#net = caffe.Net('/home/ztq/caffe/examples/myfile/lenet_train_test.prototxt',caffe.TEST)
#net = caffe.Net('/home/ztq/caffe/examples/myfile/lenet_train_test.prototxt',\
#                       '/home/ztq/caffe/examples/myfile/lenet_iter_10000.caffemodel',caffe.TEST)
'''
net = caffe.Net('/home/ztq/caffe/examples/myfile/lenet.prototxt',\
                       '/home/ztq/caffe/examples/myfile/lenet_iter_10000.caffemodel',caffe.TEST)

im = plt.imread("/home/ztq/GitWorkSpace/神经网络作业/TestImages/2.jpg") 
'''

net = caffe.Net('/home/ztq/caffe/examples/lpr_test/lenet.prototxt',\
                       '/home/ztq/caffe/examples/lpr_test/lenet_iter_10000.caffemodel',caffe.TEST)
im = plt.imread("/home/ztq/caffe/data/lpr/dataSet/testData/zh_e-70.jpg")

img = np.zeros((im.shape[0],im.shape[1],1))
img[:,:,0] = im.copy()#必须转化为3维的数组，哪怕是灰度图，最后一维也应该是1

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})#resize图像
transformer.set_transpose('data', (2,0,1))#交换array的每个维度，在这里将(20,20,1)换为(1,20,20)
transformer.set_raw_scale('data', 0.00390625)  #所有数据乘以scale
transformer.set_mean('data', np.load('/home/ztq/caffe/examples/lpr_test/meanVal0/mean.npy').mean(1).mean(1)) # 减去0均值
#transformer.set_channel_swap('data', (2,1,0))#交换图像的各个通道，在这里将RGB换为BGR，只对blob中的第二个位置，即(1,3,20,20)中的3中的通道互相交换
#net.blobs['data'].reshape(1,1,20,20)
net.blobs['data'].data[...] = transformer.preprocess('data',img)

data = net.blobs['data'].data[0][0]
plt.imshow(data)

print ''
print net.forward()
print ''
'''
print 'net.blobs.keys(): ',net.blobs.keys()
print 'data-data: ',net.blobs['data'].data.shape
print 'conv1-data: ',net.blobs['conv1'].data.shape
print 'pool1-data: ',net.blobs['pool1'].data.shape
print 'conv2-data: ',net.blobs['conv2'].data.shape
print 'pool2-data: ',net.blobs['pool2'].data.shape
print 'ip1-data: ',net.blobs['ip1'].data.shape
print 'ip2-data: ',net.blobs['ip2'].data.shape
print ' '
print 'net.params.keys(): ',net.params.keys()
print 'conv1-params: ',net.params['conv1'][0].data.shape
print 'conv2-params: ',net.params['conv2'][0].data.shape
print 'ip1-params: ',net.params['ip1'][0].data.shape
print 'ip2-params: ',net.params['ip2'][0].data.shape
'''
print [(k,v.data.shape) for k,v in net.blobs.items()]#data
print ' '
print [(k,v[0].data.shape) for k,v in net.params.items()]#weights                     
print [(k,v[1].data.shape) for k,v in net.params.items()]#biases
img_id = 0
print 'result = ',name_list[np.argmax(net.blobs['prob'].data[img_id])]
print 'prob = ',np.max(net.blobs['prob'].data[img_id])


#use .copy() to protect the data being modified!!!
mytools.show_data(net.blobs['data'].data[img_id].copy(),'data image')
mytools.show_data(net.blobs['conv1'].data[img_id].copy(),'conv1 image')
mytools.show_data(net.blobs['pool1'].data[img_id].copy(),'pool1 image')
mytools.show_data(net.blobs['conv2'].data[img_id].copy(),'conv2 image')
mytools.show_data(net.blobs['pool2'].data[img_id].copy(),'pool2 image')
plt.figure()
plt.plot(net.blobs['prob'].data[img_id])

mytools.show_feature(net.params['conv1'][0].data[:,0,:,:].copy(),'conv1 weights(filter)')
#mytools.show_feature(net.params['conv2'][0].data[:,0,:,:].copy(),'conv2 weights(filter)')
mytools.show_feature(net.params['conv2'][0].data.reshape(50*40,5,5).copy(),'conv2 weights(filter)')
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:49:03 2016

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ztq/GitWorkSpace/caffePythonTest/')
sys.path.append('/home/ztq/caffe/python/')
import caffe
import CaffePythonTools as mytools

#读网络和权重模型
caffe.set_mode_gpu()
model_def = "/home/ztq/caffe/models/bvlc_reference_caffenet/deploy.prototxt"
model_weights = "/home/ztq/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
net = caffe.Net(model_def,model_weights,caffe.TEST)

#读图片
img_src = plt.imread('/home/ztq/caffe/examples/images/cat.jpg')
plt.imshow(img_src)

#转化图片到caffe中data层所需的blob形式
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})#resize图像
transformer.set_transpose('data', (2,0,1))#交换array的每个维度，在这里将(256,256,3)换为(3,256,256)
transformer.set_mean('data', np.load('/home/ztq/caffe/data/ilsvrc12/imagenet_mean.npy').mean(1).mean(1)) # 减去均值
transformer.set_raw_scale('data', 1)  #所有数据乘以scale
transformer.set_channel_swap('data', (2,1,0))#交换图像的各个通道，在这里将RGB换为BGR，只对blob中的第二个位置，即(1,3,20,20)中的3中的通道互相交换
net.blobs['data'].data[...] = transformer.preprocess('data',img_src)

#输出预测结果
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()
labels = np.loadtxt("/home/ztq/caffe/data/ilsvrc12/synset_words.txt", str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items
print 'probabilities and labels:'
for p,l in zip(output_prob[top_inds], labels[top_inds]):
    print p ," " , l

#查看网络中的数据和参数
print "blobs data:"
for k,v in net.blobs.items():
    print k + '\t' +str(v.data.shape)
print "params weights:"
for k,v in net.params.items():
    print k + '\t' + str(v[0].data.shape) +str(v[1].data.shape)
    
#画filters
mytools.show_data(net.params['conv1'][0].data.transpose(0,2,3,1).copy(),'filters conv1')
mytools.show_data(net.params['conv2'][0].data.reshape(256*48,5,5).copy(),'filters conv2')
mytools.show_data(net.blobs['data'].data[0],'data data')
mytools.show_data(net.blobs['conv1'].data[0],'data conv1')

#画prob
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
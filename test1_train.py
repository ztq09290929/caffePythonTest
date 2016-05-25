# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 21:27:23 2016

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ztq/GitWorkSpace/caffePythonTest/')
sys.path.append('/home/ztq/caffe/python/')
import caffe
import CaffePythonTools as mytools
caffe.set_device(0)
caffe.set_mode_gpu()

#solver = caffe.SGDSolver( '/home/ztq/caffe/examples/myfile/lenet_solver.prototxt')
solver = caffe.SGDSolver( '/home/ztq/caffe/examples/lpr_test/lenet_solver.prototxt')
'''如果不需要绘制曲线，只需要训练出一个caffemodel, 直接调用solver.solve()就可以了'''
print [(k,v.data.shape) for k,v in solver.net.blobs.items()]
print [(k,v[0].data.shape) for k,v in solver.net.params.items()]
#solver.solve()

'''如果要绘制曲线，就需要把迭代过程中的值保存下来，因此不能直接调用solver.solve(), 需要迭代。在迭代过程中，每迭代200次测试一次'''

n_iter =10000#迭代次数
test_interval = 100#测试间隔
train_loss = np.zeros(int(np.ceil(n_iter / test_interval)))
test_acc = np.zeros(int(np.ceil(n_iter / test_interval)))
# the main solver loop
for it in range(n_iter):
    solver.step(1)  # SGD by Caffe  
    solver.test_nets[0].forward(start='conv1')
    
    if it % test_interval == 0:
        # store the train loss
        train_loss[it // test_interval] = solver.net.blobs['loss'].data
           
        acc=solver.test_nets[0].blobs['accuracy'].data
        print 'Iteration', it, 'testing...','accuracy:',acc
        test_acc[it // test_interval] = acc

plt.figure(1)
ax1 = plt.subplot(211)
ax1.plot(test_interval * np.arange(len(train_loss)),train_loss)
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax1.grid(True)
ax2 = plt.subplot(212)
ax2.plot(test_interval*np.arange(len(test_acc)),test_acc)
ax2.set_xlabel('iteration')
ax2.set_ylabel('test accuracy')
ax2.grid(True)
ax2.axis([0,3000,0,1.1])
plt.show()

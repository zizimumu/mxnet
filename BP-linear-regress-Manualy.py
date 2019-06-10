import numpy as np
import os
import gzip
import struct
import logging
import random
import math

import mxnet as mx
import matplotlib.pyplot as plt # 这是常用的绘图库
logging.getLogger().setLevel(logging.DEBUG)



n_sample = 1000
n_val = 20
train_in = [ 0 for c in range(n_sample)]
train_out = [0 for c in range(n_sample)]
val_in = [ 0 for c in range(n_val)]
val_out = [0 for c in range(n_val)]

pi = math.pi
w = 2*pi
interval = 1.0/n_sample
for i in range(n_sample):
    train_in[i] = i*interval
    train_out[i] = 2*math.sin(w*train_in[i])
    #train_out[i] = (train_out[i] +1)/2
    
interval = 1/n_val
for i in range(n_val):
    val_in[i] = i*interval
    val_out[i] = 2*math.sin(w*val_in[i])
    #val_out[i] = (val_out[i] +1)/2
    
    
'''
n_sample = 1000
train_in = [[ random.uniform(-0.5, 0.5) for c in range(2)] for n in range(n_sample)] 
train_out = [0 for n in range(n_sample)]
for i in range(n_sample):
    train_out[i] = train_in[i][0]+train_in[i][1]
    
    
val_in = [[ random.uniform(-0.5, 0.5) for c in range(2)] for n in range(20)] 
val_out = [0 for n in range(20)]
for i in range(20):
    val_out[i] = val_in[i][0]+val_in[i][1]
'''
    
    
    
batch_size = 1 # 批大小

# 迭代器
#train_iter = mx.io.NDArrayIter(np.array(train_in),np.array(lab_out),batch_size, shuffle = True)
train_iter = mx.io.NDArrayIter(data = np.array(train_in), label = {'reg_label':np.array(train_out)}, batch_size = batch_size, shuffle = True)
val_iter = mx.io.NDArrayIter(np.array(val_in), np.array(val_out), batch_size)  



# 设置网络
data = mx.symbol.Variable('data')
#sigmoid
fc1  = mx.sym.FullyConnected(data=data, num_hidden=10, name="fc1")
act1 = mx.sym.Activation(data=fc1, act_type="sigmoid", name="act1")

fc2  = mx.sym.FullyConnected(data=act1, num_hidden=10, name="fc2")
act2 = mx.sym.Activation(data=fc2, act_type="sigmoid", name="act2")

fc3  = mx.sym.FullyConnected(data=act2, num_hidden=1, name="fc3")
#act3 = mx.sym.Activation(data=fc3, act_type="sigmoid", name="act3")
net  = mx.sym.LinearRegressionOutput(data=fc3, name='reg')



# 我们将调用MXNet中的viz库，需要先告知MXNet输入数据的格式
shape = {"data" : (batch_size, 1, 1, 1)}
mx.viz.print_summary(symbol=net, shape=shape)


# 由于训练数据量较大，这里采用了GPU，若读者没有GPU，可修改为CPU

module = mx.mod.Module(symbol=net,context=mx.cpu(),label_names = (['reg_label']))
module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

module.init_params(initializer=mx.init.Uniform(scale=0.5)) #init weight from [-0.1 0.1]
module.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
metric = mx.metric.create('mse')

    
mse = 0.0
for epoch in range(100): # 手动训练，这里只训练1个epoch
    train_iter.reset() # 每个epoch需手动将迭代器复位
    metric.reset() # 这里希望看网络的训练细节，所以对于每个样本都将指标复位
    
    mse = 0.0 

    for batch in train_iter: # 对于每个batch...
        #metric.reset()
        module.forward(batch, is_train=True) # 前向传播
        module.update_metric(metric, batch.label) # 更新指标
        module.backward() # 反向传播，计算梯度
        module.update() # 根据梯度情况，由优化器更新网络参数
        mse = mse + metric.get()[1] 

    
    print('Epoch %d, MSE %s ' % (epoch, metric.get()))

       
        
count = 0
for j in range(0,20):
    module.forward(val_iter.next(), is_train=False)
    out = module.get_outputs()[0].asnumpy()
    print('test output is %f, actul is %f' % (out,val_out[j]))


     

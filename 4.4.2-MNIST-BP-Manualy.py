import numpy as np
import os
import gzip
import struct
import logging
import mxnet as mx
import matplotlib.pyplot as plt # 这是常用的绘图库
logging.getLogger().setLevel(logging.DEBUG)

def read_data(label_url, image_url): # 读入训练数据
    with gzip.open(label_url) as flbl: # 打开标签文件
        magic, num = struct.unpack(">II", flbl.read(8)) # 读入标签文件头
        label = np.fromstring(flbl.read(), dtype=np.int8) # 读入标签内容
    with gzip.open(image_url, 'rb') as fimg: # 打开图像文件
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16)) # 读入图像文件头，rows和cols都会是28
        image = np.fromstring(fimg.read(), dtype=np.uint8) # 读入图像内容
        image = image.reshape(len(label), 1, rows, cols) # 设置为正确的数组格式
        image = image.astype(np.float32)/255.0 # 归一化为0到1区间
    return (label, image)

# 读入数据
(train_lbl, train_img) = read_data('train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
 
batch_size = 16 # 批大小

# 迭代器
train_iter = mx.io.NDArrayIter(train_img, train_lbl, batch_size, shuffle=True)
#train_iter = mx.io.NDArrayIter(train_img,train_lbl,batch_size,label_name='reg_label',shuffle=True)
#train_iter = mx.io.NDArrayIter(data = np.array(train_img), label = {'reg_label':np.array(train_lbl)}, batch_size = batch_size, shuffle = True)

val_iter = mx.io.NDArrayIter(val_img, val_lbl, batch_size)  



# 设置网络
data = mx.symbol.Variable('data')

# 将图像摊平，例如1*28*28的图像就变为784个数据点，这样才可与普通神经元连接
flatten = mx.sym.Flatten(data=data, name="flatten")

# 第1层网络及非线性激活 128
fc1  = mx.sym.FullyConnected(data=flatten, num_hidden=128, name="fc1")
act1 = mx.sym.Activation(data=fc1, act_type="relu", name="act1")

# 第2层网络及非线性激活 64
fc2  = mx.sym.FullyConnected(data=act1, num_hidden=64, name="fc2")
act2 = mx.sym.Activation(data=fc2, act_type="relu", name="act2")
    
	

# 输出神经元，因为需分为10类，所以有10个神经元
fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10, name="fc3")



#act3 = mx.sym.Activation(data=fc3, act_type="relu", name="act3")
#net = mx.sym.LinearRegressionOutput(data = act3, name = 'reg')

# SoftMax
net  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')




# 我们将调用MXNet中的viz库，需要先告知MXNet输入数据的格式
shape = {"data" : (batch_size, 1, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)



# 由于训练数据量较大，这里采用了GPU，若读者没有GPU，可修改为CPU
#module = mx.mod.Module(symbol=net,context=mx.cpu(),label_names = (['reg_label']))
module = mx.mod.Module(symbol=net,context=mx.cpu())

print(train_iter.provide_label)

module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

module.init_params(initializer=mx.init.Uniform(scale=0.1)) #init weight from [-0.1 0.1]
module.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.03), ))
metric = mx.metric.create('mse')

    
mse = 0.0
for epoch in range(10): # 手动训练，这里只训练1个epoch
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

    
    print('Epoch %d, MSE %s , average %f' % (epoch, metric.get(),(mse/1875.0)))
    #predict_data()

       
        
count = 0
for j in range(0,int(10000/batch_size)):
    module.forward(val_iter.next(), is_train=False)
    out = module.get_outputs()[0].asnumpy().argmax(axis=1)
    for i in range(0,batch_size):
        if out[i] != val_lbl[j*batch_size + i] :
            count = count + 1

print('test done , error rate is %f' % (count/10000.0)) 

     

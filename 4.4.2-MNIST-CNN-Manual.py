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
 
batch_size = 32 # 批大小

# 迭代器
train_iter = mx.io.NDArrayIter(train_img, train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_img, val_lbl, batch_size)  




data = mx.symbol.Variable('data')

# 第1个卷积层，以及相应的BN和非线性，有32个5*5卷积
conv1 = mx.sym.Convolution(data=data, name="conv1", kernel=(5,5), num_filter=32)
# 对于BN层我们往往会加上fix_gamma=False这个参数
bn1 = mx.sym.BatchNorm(data=conv1, name="bn1", fix_gamma=False)
act1 = mx.sym.Activation(data=bn1, name="act1", act_type="relu")
# 第1个池化层，使用了3*3大小和2*2步长
pool1 = mx.sym.Pooling(data=act1, name="pool1", pool_type="max", kernel=(3,3), stride=(2,2))

# 第2个卷积层，以及相应的BN和非线性，有64个5*5卷积
conv2 = mx.sym.Convolution(data=pool1, name="conv2", kernel=(5,5), num_filter=64)
bn2 = mx.sym.BatchNorm(data=conv2, name="bn2", fix_gamma=False)
act2 = mx.sym.Activation(data=bn2, name="act2", act_type="relu")
# 第2个池化层，使用了3*3大小和2*2步长
pool2 = mx.sym.Pooling(data=act2, name="pool2", pool_type="max", kernel=(3,3), stride=(2,2))

# 第3个卷积层，有10个3*3卷积
conv3 = mx.sym.Convolution(data=pool2, name="conv3", kernel=(3,3), num_filter=10)
# 第3个池化层，这里设置global_pool进行全局池化，会忽略kernel的值
pool3 = mx.sym.Pooling(data=conv3, name="pool3", global_pool=True, pool_type="avg", kernel=(1,1))
# 将图像摊平，这里的效果是将10张1*1的图像摊平，因此得到10个数
flatten = mx.sym.Flatten(data=pool3, name="flatten")
# SoftMax层，将10个数变为10个分类的概率
net = mx.sym.SoftmaxOutput(data=flatten, name='softmax')


# 我们将调用MXNet中的viz库，需要先告知MXNet输入数据的格式
shape = {"data" : (batch_size, 1, 28, 28)}
mx.viz.print_summary(symbol=net, shape=shape)



# 由于训练数据量较大，这里采用了GPU，若读者没有GPU，可修改为CPU

module = mx.mod.Module(symbol=net,context=mx.cpu())
module.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

module.init_params(initializer=mx.init.Uniform(scale=0.1)) #init weight from [-0.1 0.1]
module.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.03), ))
metric = mx.metric.create('mse')

mse = 0.0
for epoch in range(10): # 手动训练，这里只训练1个epoch
    train_iter.reset() # 每个epoch需手动将迭代器复位
    metric.reset() # 这里希望看网络的训练细节，所以对于每个样本都将指标复位
    
    mse = 0.0 
    # 实际训练时，应在此调用 metric.reset() 将性能指标复位
    for batch in train_iter: # 对于每个batch...
        metric.reset()
        module.forward(batch, is_train=True) # 前向传播
        module.update_metric(metric, batch.label) # 更新指标
        module.backward() # 反向传播，计算梯度
        module.update() # 根据梯度情况，由优化器更新网络参数
        mse = mse + metric.get()[1] 

    
    print('Epoch %d, MSE %s , average %f' % (epoch, metric.get(),(mse/1875.0)))
       
        
count = 0


for j in range(0,int(10000/batch_size)):
    module.forward(val_iter.next(), is_train=False)
    out = module.get_outputs()[0].asnumpy().argmax(axis=1)
    for i in range(0,batch_size):
        if out[i] != val_lbl[j*batch_size + i] :
            count = count + 1

print('test done , validate rate is %f' % (count/10000.0))        

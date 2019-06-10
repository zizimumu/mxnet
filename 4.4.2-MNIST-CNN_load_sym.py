import numpy as np
import os
import gzip
import struct
import logging
import time
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
# (train_lbl, train_img) = read_data('train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
 
new_batch_size = 1

# 迭代器
val_iter = mx.io.NDArrayIter(val_img, val_lbl, new_batch_size)  



# load model
model_prefix = 'MNIST_symb.params'

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 10)

model = mx.mod.Module(symbol=sym)
model.bind(for_training=False, data_shapes=[("data", (new_batch_size, 1, 28, 28))])
model.set_params(arg_params, aux_params)

val_iter.reset()
# predict


time_start = time.time()
predict_stress = model.predict(val_iter, 1000)
time_end = time.time()
print("forward time is %f" % ((time_end-time_start)/1000)) # the speed in PC is about 0.000315s/per forward
    
    
out = predict_stress.asnumpy().argmax(axis=1)
for i in range(0,1000):
    if out[i] != val_lbl[i] :
        print('%d is recognizy to %d',val_lbl[i],out[i])

#print (predict_stress) # you can transfer to numpy array
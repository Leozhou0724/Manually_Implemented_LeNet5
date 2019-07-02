'''
ECE 579
Machine Learning Final Project
Yuhang Zhou
yz853
5/8/2019

Type-A Project
It's a manually implemented LeNet-5 Neural Network for the MNIST dataset
The MNIST dataset is already included in the floder which is the "mnist.pkl" file
All the layer classes are saved in "layers.py" file

Reference: Build Lenet from Scratch - Gary
	       https://medium.com/deep-learning-g/build-lenet-from-scratch-7bd0c67a151e
		   Convolutional Neural Network implemenation from scratch in python numpy - vaibhavnaggar
		   https://github.com/vaibhavnaagar/cnn
		   lenet-5-mnist-from-scratch-numpy - toxtli
		   https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy

'''

import numpy as np
import pickle
import time
from layers import conv, fc, max_pool, relu, softmax


# loss function
def cross_entropy_loss(y_pred, y):
    prob = np.sum(y.reshape(1, y.shape[0]) * y_pred)
    loss = -np.log(prob)
    return loss

# LeNet-5 class


class LeNet5:
    def __init__(self):
        self.lr = 0.01
        # conv net
        self.c1 = conv(1, 6, kernel=5, learning_rate=self.lr)
        self.relu1 = relu()
        self.s2 = max_pool(kernel=2, stride=2)
        self.c3 = conv(6, 16, kernel=5, learning_rate=self.lr)
        self.relu3 = relu()
        self.s4 = max_pool(kernel=2, stride=2)
        self.c5 = conv(16, 120, kernel=4, learning_rate=self.lr)
        self.relu5 = relu()
        # fc net
        self.f6 = fc(120, 84, learning_rate=self.lr)
        self.relu6 = relu()
        self.f7 = fc(84, 10)
        self.sig7 = softmax()
        # record the shape between the conv net and fc net
        self.conv_out_shape = None

    def forward(self, X):
        out = self.c1.forward(X)

        out = self.relu1.forward(out)
        out = self.s2.forward(out)
        out = self.c3.forward(out)
        out = self.relu3.forward(out)
        out = self.s4.forward(out)
        out = self.c5.forward(out)
        out = self.relu5.forward(out)
        self.conv_out_shape = out.shape
        out = out.reshape(1, -1)
        out = self.f6.forward(out)
        out = self.relu6.forward(out)
        out = self.f7.forward(out)
        out = self.sig7.forward(out)
        return out

    def backward(self, dout):
        dout = self.sig7.backward(dout)
        dout = self.f7.backward(dout)
        dout = self.relu6.backward(dout)
        dout = self.f6.backward(dout)
        dout = dout.reshape(self.conv_out_shape)
        dout = self.relu5.backward(dout)
        dout = self.c5.backward(dout)
        dout = self.s4.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.c3.backward(dout)
        dout = self.s2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.c1.backward(dout)


# load data from mnist.kpl file
f = open('mnist.pkl', 'rb')
data = pickle.load(f)
x = data['training_images']
y = data['training_labels']
x_test = data['test_images']
y_test = data['test_labels']
x = x.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
y = np.eye(10)[y]
y_test = np.eye(10)[y_test]
print('Training data size:', x.shape)
print('Training label size:', y.shape)
model = LeNet5()
epoch = 10
batch_size = 20
t0 = time.time()
for i in range(epoch):
    for batch_idx in range(1, x.shape[0], batch_size):
        if batch_idx + batch_idx < x.shape[0]:  # not the last batch
            x_batch = x[batch_idx:batch_idx + batch_size]
            y_batch = y[batch_idx:batch_idx + batch_size]
        else:
            x_batch = x[batch_idx:batch_idx + batch_size]
            y_batch = y[batch_idx:batch_idx + batch_size]
            # the left data of the last batch may be not enought for a full batch

        loss = 0
        acc = 0
        for n in range(batch_size):
            x_tmp = x_batch[n]
            y_tmp = y_batch[n]
            y_pred = model.forward(x_tmp)

            loss += cross_entropy_loss(y_pred, y_tmp)
            if np.argmax(y_pred) == np.argmax(y_tmp):
                acc += 1
            dout = y_tmp
            # print(y_tmp)
            model.backward(dout)
        loss /= batch_size
        t1 = time.time()
        print('Epoch:{} Batch:{}    ({:.6f}%)    Loss:{:.6f}    Runtime:{:.2f}s'.format(
            i, batch_idx, (batch_idx / x.shape[0]) * 100, loss, t1 - t0))

# test
acc = 0
loss = 0
for n in range(x_test.shape[0]):
    x_tmp = x_test[n]
    y_tmp = y_test[n]
    y_pred = model.forward(x_tmp)

    loss += cross_entropy_loss(y_pred, y_tmp)
    if np.argmax(y_pred) == np.argmax(y_tmp):
        acc += 1
    dout = y_tmp
    model.backward(dout)
loss /= batch_size
print('Test result: final loss={:.6f}, accuracy={:.6f}'.format(loss,acc/y_test.shape[0]))
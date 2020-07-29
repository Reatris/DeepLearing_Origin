#2层神经网络尝试
import numpy as np

import min_batch_cross_entropy_error as Min_batch_cross_entropy_error
import numerical_gradient as Numerical_gradient
import sigmoid as Sigmoid
import softmax as Softmax


class TowLayerNet:
    #分别是 输入信号个数 隐藏感知机个数输出信号个数批处理个数
    def __init__(self,input_size,hidden_size,output_size,batch_size,weight_init_std=0.01):
        self.params={}
        self.params['w1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros((batch_size,hidden_size))
        self.params['w2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros((batch_size,output_size))
        self.params['batch_size']=batch_size

    def predict(self,x):
        w1,w2=self.params['w1'],self.params['w2']
        b1,b2=self.params['b1'],self.params['b2']

        if x.ndim==3:#重新编排4维数组形状
            x=np.reshape(x,(b1.shape[0],w1.shape[0]))

        a1=np.dot(x,w1)+b1
        #两层神经元之间用sigmoid函数连接
        z1=Sigmoid.sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        y=Softmax.softmax(a2,self.params['batch_size'])#如果速batch运算必须倍化

        return y

    #x:输入数据,t:监督函数
    def loss(self,x,t):
        y=self.predict(x)

        #用了批处理所以要用min_batch版损失函数
        return Min_batch_cross_entropy_error.cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)#获取y数组里面的最大值索引（是个数组 ，axis=0是列，=1是行）
        t=np.argmax(t,axis=1)
        
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    #x：输入数据，t：监督数据
    def numerical_gradient(self,x,t):
        loss_W=lambda W:self.loss(x,t)

        grads={}
        grads['w1']=Numerical_gradient.numerical_gradient(loss_W,self.params['w1'])
        grads['b1']=Numerical_gradient.numerical_gradient(loss_W,self.params['b1'])
        grads['w2']=Numerical_gradient.numerical_gradient(loss_W,self.params['w2'])
        grads['b2']=Numerical_gradient.numerical_gradient(loss_W,self.params['b2'])

        return  grads


import numpy as np
from collections import  OrderedDict
import affine as  Affine
import softmaxwithloss as Softmaxwithloss
import numerical_gradient as Numerical_gradient
import relu as Relu
import Dropout as dropout

class  newtwolayernet:

    def __init__(self,input_size,hidden_size,output_size,batch_size=100):
        #初始化权重 
        self.params={}
        self.params['w1']=(np.sqrt(2/50))*np.random.randn(input_size,hidden_size)
        #下一层链接ReLU所以权重标准差用He初始化np.sqrt(2/50)
        self.params['b1']=np.zeros((batch_size,hidden_size))
        self.params['w2']=0.01*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros((batch_size,output_size))
        self.batch_size=batch_size
        #生成层
        self.layers=OrderedDict()#有序字典存放各层神经节点
        self.layers['Affine1']=Affine.affine(self.params['w1'],self.params['b1'])
        self.layers['Dropout']=dropout.Dropout(dropout_ratio=0.5)#droout层
        self.layers['Relu1']=Relu.relu()
        self.layers['Affine2']=Affine.affine(self.params['w2'],self.params['b2'])
        self.lastlayer=Softmaxwithloss.softmaxwithloss()

    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)#开始正向传播

        return x

    # x:输入数据，t：监督数据
    def loss(self,x,t):
        y=self.predict(x)
        return self.lastlayer.forward(y,t,self.batch_size)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)#取行方向上的最大值，此处为最大概率的索引标签
        if  t.ndim !=1:t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    #x：输入数据，t：监督数据
    def numerical_gradient(self,x,t):
        loss_w=lambda w:self.loss(x,t)
        grads={}
        grads['w1']=Numerical_gradient.numerical_gradient(loss_w,self.params['w1'])
        grads['b1']=Numerical_gradient.numerical_gradient(loss_w,self.params['b1'])
        grads['w2']=Numerical_gradient.numerical_gradient(loss_w,self.params['w2'])
        grads['b2']=Numerical_gradient.numerical_gradient(loss_w,self.params['b2'])

        return grads

    def gradient(self,x,t):
        #forward 过一遍所有层数
        self.loss(x,t)

        #backward
        dout=1
        #dout.shape是(100,10)
        dout=self.lastlayer.backward(dout)#返回的是正确率关系数据
        layers=list(self.layers.values())#反向排序层数
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)

        #设定 
        grads={}
        grads['w1']=self.layers['Affine1'].dw
        grads['b1']=self.layers['Affine1'].db
        grads['w2']=self.layers['Affine2'].dw
        grads['b2']=self.layers['Affine2'].db
        
        return grads
        




        



        
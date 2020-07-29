import numpy as np
from collections import  OrderedDict
import Convolution as convolution
import relu as Relu
import Pooling as pooling
import affine as Affine
import softmaxwithloss as Softmaxwithloss
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),conv_param={'filter_num':30, 'filter_size':5,'pad':0, 'stride':1},hidden_size=100, output_size=10, weight_init_std=0.01,batch_size=100):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad / filter_stride + 1)
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0],filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std *np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] =convolution.Convolution(self.params['W1'],self.params['b1'],conv_param['stride'],conv_param['pad'])

        self.layers['Relu1'] = Relu.relu()
        self.layers['Pool1'] = pooling.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine.affine(self.params['W2'],self.params['b2'])

        self.layers['Relu2'] = Relu.relu()
        self.layers['Affine2'] = Affine.affine(self.params['W3'],self.params['b3'])
        self.lastLayer = Softmaxwithloss.softmaxwithloss()

        self.batch_size=batch_size
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t,self.batch_size)

    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)#取行方向上的最大值，此处为最大概率的索引标签
        if  t.ndim !=1:t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dw
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dw
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dw
        grads['b3'] = self.layers['Affine2'].db

        return grads
import numpy as np
class affine:
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.x=None
        self.dw=None
        self.db=None
        self.original_x_shape = None#池化层原来的形状

    def forward(self,x):
        self.original_x_shape = x.shape
        x=x.reshape(100,-1)
        self.x=x
        out=np.dot(x,self.w)+self.b

        return out

    def backward(self,dout):
        dx=np.dot(dout,self.w.T)
        #w要转置w.shape原来是50，10,转置后就是(100,10)x(10,50)=(100,50)
        self.dw=np.dot(self.x.T,dout)
        #反向传播，原理同上
        self.db=np.sum(dout,axis=0)
        #这里(100,10)==>(1,10)

        dx=dx.reshape(*self.original_x_shape)#还原形状

        return dx
        




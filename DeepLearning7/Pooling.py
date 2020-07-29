import im2col as Im2col
import col2im as Col2im
import numpy as np
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride = stride
        self.pad = pad
        
    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1+(H-self.pool_h)/self.stride)
        out_w = int(1+(W-self.pool_w)/self.stride)

        #展开（1）
        col = Im2col.im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)
        
        #最大值（2）
        out = np.max(col,axis=1)
        #取最大值坐标反向传播用
        arg_max = np.argmax(col,axis=1)
        self.x = x
        self.arg_max = arg_max
        #转换（3）
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out

    def backward(self,dout):
        #dout_size = dout.shape[0]#传回的是二维
        
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size,pool_size))#别忘记了双层
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx =Col2im.col2im(dcol, self.x.shape,self.pool_h, self.pool_w,self.stride,self.pad)

        return dx



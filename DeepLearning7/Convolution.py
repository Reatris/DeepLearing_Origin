import im2col as Im2col
import col2im as Col2im
import numpy as np
class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad
    
    def forward(self,x):
        FN,C,FH,FW=self.w.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)

        col=Im2col.im2col(x,FH,FW,self.stride,self.pad)
        #得到  （ OH*OW*N ，FH*FW*C）的数组，之后和滤波器相乘。
        col_w=self.w.reshape(FN,-1).T 
        #滤波器展开,-1为自动调整的值即FN*-1<==>FN*C*FW*FH
        out=np.dot(col,col_w)+self.b

        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
        #这部实际上是把数据维度调回原来的
        self.x=x
        self.col=col
        self.col_w=col_w

        return out

    def backward(self,dout):
        (FN,C,FH,FW)=self.w.shape
        dout=dout.transpose(0,2,3,1).reshape(-1,FN)
        #倒过来重新做

        self.db = np.sum(dout,axis=0)
        self.dw = np.dot(self.col.T,dout)
        self.dw = self.dw.transpose(1,0).reshape(FN,C,FH,FW)
        #运算完毕之后转回(C,FN,FH,FW)
        dcol = np.dot(dout,self.col_w.T)
        dx = Col2im.col2im(dcol,self.x.shape,FW,FH,self.stride,self.pad)
        return dx

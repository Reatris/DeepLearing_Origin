class Convolution:
    # 初始化
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        (FN, C, FH, FW) = self.W.shape
        (N, C, H, W) = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1) 
        #reshape函数会自动计算-1维度上的元素个数，以使多维数组的元素个数前后一致
        out = np.dot(col, col_W) + self.b
        
        out = out.reshape(N, out_h,out_w,-1).transpose(0, 3, 1, 2)
        # transpose会更改多维数组的轴的顺序。
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
        
    def backword(self,dout):
        (FN,C,FH,FW) = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,FN)
        # 改变损失中数组的轴的顺序，再用reshape转为二维数组
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1,0).reshape(FN,C,FH,FW)
        # 用reshape转为二维数组，再改变损失中数组的轴的顺序
        
        dcol = np.dot(dout,self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx
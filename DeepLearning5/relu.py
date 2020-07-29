class relu:
    def __init__(self):
        self.mask=None

    def forward(self,x):
        self.mask=(x<=0)#将数组里面小于0的数组下标加入mask
        out=x.copy()
        out[self.mask]=0

        return out

    def  backward(self,dout):#dout.shape==>(100,50)
        dout[self.mask]=0
        dx=dout #dx.shape==>(100,50)
        return dx

    


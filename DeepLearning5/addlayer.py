#反向传播加法层
class addlayer:
    def __init__(self):
        pass#什么都不做，跳过

    def forward(self,x,y):
        out=x+y
        return out

    def backward(self,dout):
        dx = dout*1
        dy=dout*1
        return dx,dy
        
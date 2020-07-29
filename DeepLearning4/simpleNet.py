import softmax as  softmax
import numpy as np
import cross_entropy_error as cee
import numerical_gradient as ng

class simpleNet:
    def __init__(self):
        #用高斯分布进行初始化产生一个2*3的numpy数组值为随机[0,1]
        self.W=np.random.randn(2,3)
        
    def predict(self,x):
        return  np.dot(x,self.W)

    def loss(self,x,t):
        z=self.predict(x)
        y=softmax.softmax(z)
        loss=cee.cross_entropy_error(y,t)
        return loss
    


def  f(W):#返回损失函数 用来计算参数，这里w是 伪参
        return net.loss(x,t)
if  __name__=='__main__':
    #创建一个simplenet
    net=simpleNet()
    #打印权重参数
    print('权重参数为'+str(net.W))
    print('输入为x=np.array([0.6,0.9])')
    x=np.array([0.6,0.9])
    p=net.predict(x)
    print('得出预测概率为'+str(p))
    maxp=np.argmax(p)
    print('最大索引是'+str(maxp))
    #设置 目标正解标签
    t=np.array([0,0,1])
    lost=net.loss(x,t)
    print('目标正解标签([0,0,1])所得损失函数是'+str(lost))
    dw=ng.numerical_gradient(f,net.W)
    print('计算所得梯度为'+str(dw))



import sys
sys.path.append('..')
import min_batch_cross_entropy_error as Min_batch_cross_entropy_error
import softmax as Softmax
class softmaxwithloss:
    def  __init__(self):
        self.loss=None #损失
        self.y=None #softmax的 输出
        self.t=None  #监督 数据 （one-hot）

    def forward(self,x,t,batch_size):
        self.t=t
        self.y=Softmax.softmax(x,batch_size)
        self.loss=Min_batch_cross_entropy_error.cross_entropy_error(self.y,self.t)

        return self.loss#这是单个标准loss而不是整个batch_size的

    def backward(self,dout=1):
        #batch_size=self.t.shape[0]
        dx=(self.y-self.t) #/batch_size

        return  dx
        #dx是预测正确率和标准的关系
        

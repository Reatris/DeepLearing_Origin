import numpy as np
def cross_entropy_error(y,t):
    if y.ndim==1:#判断y的维度
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    #delta=1e-7  #防止log0出现无穷值
    batch_size=y.shape[0]#获取y中0行的元素的总数
    return -np.sum(t*np.log(y+1e-7))/batch_size

if __name__ == '__main__':
    print("若y是多维数组（一个mini——batch组）")
    print('y=[[0.1,0.3,0.6],[0.2,0.2,0.6]]')
    print('t=[[0,0,1],[0,0,1]]')
    y=[[0.1,0.3,0.6],[0.2,0.2,0.6]]
    Y=np.array(y)#必须是numpy数组
    t=[[0,0,1],[0,0,1]]
    target=cross_entropy_error(Y,t)
    print('计算获得平均误差率（损失函数）为'+str(target))
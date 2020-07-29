import numpy as np
def cross_entropy_error(y,t):
    delta=1e-7  #防止log0出现无穷值
    return -np.sum(t*np.log(y+delta))

if __name__ =='__main__':
    print('测试解为y=[0.1,0.5,0.4] 正确解为t=[0,1,0]')
    y=np.array([0.1,0.5,0.4])
    t=np.array([0,1,0])
    target=cross_entropy_error(y,t)
    print('误差率（损失函数）为：'+str(target))
    print('测试解为y2=[0.01,0.98,0.01] 正确解为t2=[0,1,0]')
    y2=np.array([0.01,0.98,0.01])
    t2=np.array([0,1,0])
    target2=cross_entropy_error(y2,t2)
    print('误差率（损失函数）为：'+str(target2))

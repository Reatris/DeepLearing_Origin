import numpy as np
def mean_squared_error(y,t):#y是测试数据概率，t是真确概率
    return 0.5*np.sum((y-t)**2)

if __name__ =='__main__':
    print('y=[0.1,0.05,0.1]而t=[0,1,0]  (这里2位正解)')
    y=np.array([0.1,0.05,0.1])
    t=np.array([0.0,1.0,0.0])
    target=mean_squared_error(y,t)
    print('误差率为（损失函数）：'+str(target))
    print('y=[0.01,0.98,0.01]而t=[0,1,0]  (这里2位正解)')
    y2=np.array([0.01,0.98,0.01])
    t2=np.array([0.0,1.0,0.0])
    target2=mean_squared_error(y2,t2)
    print('误差率为（损失函数）：'+str(target2))


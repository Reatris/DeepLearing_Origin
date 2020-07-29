#定义一个sigmoid函数并且输入[-1.0,1.0,2.0]求得返回值

import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))



if __name__ =='__main__':
    y=np.array([-1.0,1.0,2.0])
    print(sigmoid(y))
    #现在画出sigmoid函数的图像
    import matplotlib.pylab as plt
    x=np.arange(-5.0,5.0,0.1)
    y=sigmoid(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()

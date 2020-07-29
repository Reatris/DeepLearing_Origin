import numpy as np
import matplotlib.pyplot as plt
 

def sigmoid(x):
     return 1/(1+np.exp(-x))

x=np.random.randn(1000,100)     #1000数据
node_num=100    #各隐藏层的节点（神经元）数
hidden_layer_size=5 #隐藏层有5层
activations={}  #激活层的结果保留在这

for i in range(hidden_layer_size):
    if i!=0:
        x=activations[i-1]

    #w=np.random.randn(node_num,node_num)*1 
    #权重标准差过大向两边分布==>梯度消失
    #w=np.random.randn(node_num,node_num)*0.01
    #权重标准差 改为0.01可集中分布，但是表现力受限，没有广度
    w=np.random.randn(node_num,node_num)/np.sqrt(node_num)
    #使用Xavier 标准差改为1/n n为上一层神经节点个数的sqrt
    #替换后广度增加，将sigmoid函数替换为tanh（双曲线）表现完美
    #如果使用ReLU函数做激活函数，标准差更改为sqtr(2/n)最佳

    z=np.dot(x,w)
    a=sigmoid(z)    #sigmoid函数
    activations[i]=a

#绘制直方图
for i,a in activations.items():
    plt.subplot(1,len(activations),i+1)     #行数，列数，位置
    plt.title(str(i+1)+'-layer')
    x=a.flatten()
    plt.hist(x,bins=30,range=(0,1))
    #hist参数x是一维数据，bin条形数，range x轴的范围    
    # 见https://www.jianshu.com/p/273e28cafe85
plt.show()

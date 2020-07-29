#将minst数组预处理

#将数组归一化[0,1]
import numpy as np

#将数组中的每个值除以255 将数组归一化[0,1]
def normalize(array):
    return array/255

    
def one_hot(array):
    cols=len(array)#获取行数
    temp=np.zeros((cols,10))
    for i in range(0,cols):
        j=int(array[i])
        temp[i][j]=1.0
        
    return temp

def convolutiondo(array):
    return array.reshape(1,60000,28,28).transpose(1,0,2,3)
'''
def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t
'''

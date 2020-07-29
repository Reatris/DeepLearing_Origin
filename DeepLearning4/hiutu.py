import numpy as np
import matplotlib.pyplot as plt
import random

picture={}#控字典
p=[]#空列表，用来放数值
s=[]#空列表，用来放p数组


#字典添加图像
for i in range(10):
    #采集625个像素点
    p.clear()#清空列表
    for j in range(625):
        fig=random.randint(0,255)
        p.append(fig)
    #转换列表为numpy数组用来变型   
    temp=np.array(p) 
    s.append(temp.reshape((25,25)))
    #放进去
    name='图像'+str(i)
    picture.update({name:s[i]})
#图像显示
for key,value  in picture.items():
    print('正在绘制'+key)
    plt.imshow(value)
    plt.show()


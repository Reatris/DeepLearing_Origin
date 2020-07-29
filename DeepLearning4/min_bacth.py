import os,sys
sys.path.append('..')
from DeepLearning3 import minst #沙雕报错，不用管
import matplotlib.pyplot as plt
import numpy as np

train_images=minst.get_train_images()
train_labels=minst.get_train_lables()
print(train_images.shape)
#随机选取10个图像
select=np.random.choice(60000,20)
for i in select:
        #显示图像
        plt.imshow(train_images[i],cmap='gray')
        print('正在显示图像'+str(i))
        print('标签为'+str(train_labels[i]))
        #plt.pause(0.5)
        plt.show()
        
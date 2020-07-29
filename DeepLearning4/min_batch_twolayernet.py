import numpy as np
import os,sys
sys.path.append('..')
import minst as Minst
import Towlayernet as towlayernet
import  preprocessing as Preprocessing
import random

import matplotlib.pylab as plt

train_images=Preprocessing.normalize(Minst.get_train_images())
train_lables=Preprocessing.one_hot(Minst.get_train_lables())
test_images=Preprocessing.normalize(Minst.get_test_images())
test_lables=Preprocessing.one_hot(Minst.get_test_lables())

#超参数
iters_num=120  #100次计算需要1分钟+
batch_size=100
learning_rate=0.01

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
#参数监督
dw1_list=[]
db1_list=[]
dw2_list=[]
db2_list=[]

train_size=train_images.shape[0]
test_size=test_images.shape[0]
#平均每个epoch的重复次数
iter_per_epoch=max(train_size/batch_size,1)




network=towlayernet.TowLayerNet(input_size=784,hidden_size=50,output_size=10,batch_size=100)

for i in range(iters_num):
    #获取mini_batch
    batch_mask=np.random.choice(train_size,batch_size)#返回一个数组
    train_images_batch=train_images[batch_mask]
    train_lables_batch=train_lables[batch_mask]
    
    
    #计算梯度
    grad=network.numerical_gradient(train_images_batch,train_lables_batch)
    dw1_list.append(np.mean(grad['w1']))
    db1_list.append(np.mean(grad['b1']))
    dw2_list.append(np.mean(grad['w2']))
    db2_list.append(np.mean(grad['b2']))
    #下面为高速版
    #grad=network.gradient(train_images_batch,train_lables_batchh)

    #更新参数
    for key in ('w1','b1','w2','b2'):
        if key=='w1':
            network.params[key]-=(learning_rate*grad[key])*1000
        else:
            network.params[key]-=learning_rate*grad[key]

    #记录学习过程,loss用来存放损失 函数
    loss=network.loss(train_images_batch,train_lables_batch)
    train_loss_list.append(loss)
    #计算每个epoch的识别精度
    #判断是达成一个epoch（设定600一次）
    if i % iter_per_epoch == 0:
        #计算训练数据正确率
        train_acc=network.accuracy(train_images_batch,train_lables_batch)
        #计算测试数据正确率，过拟合？
        batch_mask=np.random.choice(test_size,batch_size)#返回一个数组
        test_images_batch=test_images[batch_mask]
        test_lables_batch=test_lables[batch_mask]
        test_acc=network.accuracy(test_images_batch,test_lables_batch)
        #统计正确率
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('阶段训练数据正确率'+str(train_acc))
        print('阶段测试数据正确率'+str(test_acc))





y_loss=train_loss_list#损失函数数据
x_loss=np.arange(iters_num)#生成100个元素的1*100数组做x轴
x=x_loss#损失函数x轴
y_train_acc=train_acc_list#训练数据正确率
y_test_acc=test_acc_list#测试数据正确率
x_acc=np.arange(0,112,600)#x_acc=[0,600,1200]
plt.title(str(iters_num)+'times learning picture')
plt.plot(x,y_loss,color='red',label='loss')
plt.plot(x_acc,y_train_acc,color='green',label='train_acc')
plt.plot(x_acc,y_test_acc,color='blue',label='test_acc')
plt.plot(x,dw1_list,color="yellow",label='dw1')
plt.plot(x,db1_list,color='orange',label='db1')
plt.plot(x,dw2_list,color='black',label='dw2')
plt.plot(x,db2_list,color='pink',label='db2')

plt.legend()#显示图列
plt.xlabel('learing by times')
plt.ylabel('accuarcy  and loss')
plt.show()


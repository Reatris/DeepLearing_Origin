import sys
sys.path.append('..')
import numpy as np
import minst as Minst
import newtowlayernet as Newtwolayernet
import preprocessing as Preprocessing
import random

import matplotlib.pylab as plt

#读入数据  
train_images=Preprocessing.normalize(Minst.get_train_images())
train_lables=Preprocessing.one_hot(Minst.get_train_lables())
test_images=Preprocessing.normalize(Minst.get_test_images())
test_lables=Preprocessing.one_hot(Minst.get_test_lables())

#超参数
iters_num=1000
train_size=train_images.shape[0]
test_size=test_images.shape[0]
batch_size=10
learning_rate=0.002
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
#存放梯度 监控 
w1_list=[]
b1_list=[]
w2_list=[]
b2_list=[]
'''
#存放参数监控
b2_value_list=[]
b1_value_list=[]
w1_value_list=[]
w2_value_list=[]
value={'w1':w1_value_list,'w2':w2_value_list,'b1':b1_value_list,'b2':b2_value_list}
'''#现在这个没用了

iter_per_epoch=max(train_size/batch_size,1)

network=Newtwolayernet.newtwolayernet(input_size=784,hidden_size=50,output_size=10,batch_size=100)
for i in range(iters_num):
    batch_mask=np.random.choice(train_size,batch_size)
    train_images_batch=train_images[batch_mask]
    train_lables_batch=train_lables[batch_mask]

    #通过 误差 反向传播算法求 梯度 
    grad=network.gradient(train_images_batch,train_lables_batch)
    w1_list.append(np.mean(grad['w1']))
    b1_list.append(np.mean(grad['b1']))
    w2_list.append(np.mean(grad['w2']))
    b2_list.append(np.mean(grad['b2']))

    #更 新
    for key in ('w1','b1','w2','b2'):
        #value[key].append(np.mean(network.params[key]))#记录值
        network.params[key]-=learning_rate*grad[key]
        

    loss=network.loss(train_images_batch,train_lables_batch)
    train_loss_list.append(loss)

    if i%iter_per_epoch==0:
        batch_mask=np.random.choice(test_size,batch_size)
        test_images_batch=test_images[batch_mask]
        test_lables_batch=test_lables[batch_mask]
        train_acc=network.accuracy(train_images_batch,train_lables_batch)
        test_acc=network.accuracy(test_images_batch,test_lables_batch)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train_acc:'+str(train_acc))
        print("test_acc:"+str(test_acc))


y_loss=train_loss_list#损失函数数据
x_loss=np.arange(iters_num)#生成100个元素的1*100数组做x轴
x=x_loss#损失函数x轴
y_train_acc=train_acc_list#训练数据正确率
y_test_acc=test_acc_list#测试数据正确率
x_acc=np.arange(0,iters_num+1,600)#x_acc=[0,600,1200]
plt.title(str(iters_num)+'times learning picture')
plt.plot(x,y_loss,color='red',label='loss')
#梯度情况
plt.plot(x,w1_list,color='black',label='w1_d')
plt.plot(x,b1_list,color='yellow',label='b1_d')
plt.plot(x,w2_list,color='orange',label='w2_d')
plt.plot(x,b2_list,color='green',label='b2_d')
'''#参数情况
plt.plot(x,value['w1'],color='black',label='w1')
plt.plot(x,value['b1'],color='yellow',label='b1')
plt.plot(x,value['w2'],color='orange',label='w2')
plt.plot(x,value['b2'],color='green',label='b2')
'''

plt.plot(x_acc,y_train_acc,color='pink',label='train_acc')
plt.plot(x_acc,y_test_acc,color='blue',label='test_acc')
plt.legend()#显示图列
plt.xlabel('learing by times')
plt.ylabel('accuarcy  and loss')
plt.show()

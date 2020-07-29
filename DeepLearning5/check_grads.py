#测试微分法和 误差反向传播法 求得结果的误差
import sys
sys.path.append('..')
from DeepLearning3 import minst as Minst
import preprocessing as Preprocessing
import newtowlayernet as Newtowlayernet
import numpy  as np

#读入数据
train_images=Preprocessing.normalize(Minst.get_train_images())
train_lables=Preprocessing.one_hot(Minst.get_train_lables())
test_images=Preprocessing.normalize(Minst.get_test_images())
test_lables=Preprocessing.one_hot(Minst.get_test_lables())

network=Newtowlayernet.newtwolayernet(input_size=784,hidden_size=50,output_size=10,batch_size=990,weight_init_std=0.01)

train_images_batch=train_images[:990]
train_lables_batch=train_lables[:990]

grad_numerical=network.numerical_gradient(train_images_batch,train_lables_batch)
grad_backprop=network.gradient(train_images_batch,train_lables_batch)

#求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    print(key+':')
    print(grad_numerical[key].shape)#微分梯度结果结构
    print(grad_backprop[key].shape)#反传梯度结果结构
    diff =np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+':'+str(diff))

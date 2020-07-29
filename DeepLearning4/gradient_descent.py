#梯度下降法by wc
import numpy as np
import numerical_gradient as nmg
import numerical_diff as  nmd
import matplotlib.pylab as plt


#监督参数 
list_of_grad_x1=[]#存放梯度函数
list_of_grad_x2=[]#存放梯度函数
list_of_x1=[]#x1的值
list_of_x2=[]#x1的值
list_of_value=[]#y的值

step_num=100
#f为要优化的函数，init_x为当前起始位置（numpy.array数组），
# lr为学习率（度），step_num为 学习步长，默认设置0.01，100
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x

    for i in range(step_num):
        list_of_x1.append(x[0])
        list_of_x2.append(x[1])
        list_of_value.append(x[0]**2+x[1]**2)
        grad=nmg.numerical_gradient(f,x)
        list_of_grad_x1.append(grad[0])
        list_of_grad_x2.append(grad[1])
        x -=lr*grad

        

    return  x

if __name__ == '__main__':
    print('现在测试用梯度下降法寻找函数 f(x)=x[1]**2+x[2]**2 的最小值')
    print('梯度下降法   学习率0.05    迭代次数100')
    target=gradient_descent(nmd.function_2,np.array([2.0,-1.0]),0.05,100)
    print('最小值处为'+str(target))
    plt.title('grand_descent with times')
    plt.ylabel('value')
    plt.xlabel('times')
    x_times=np.arange(step_num)#x轴
    #监督参数显示
    plt.plot(x_times,list_of_grad_x1,color="red",label='grad_x1')
    plt.plot(x_times,list_of_grad_x2,color="yellow",label='grad_x2')
    plt.plot(x_times,list_of_x1,color="blue",label='x1')
    plt.plot(x_times,list_of_x2,color="pink",label='x2')
    plt.plot(x_times,list_of_value,color="gray",label='value=y')


    plt.legend()#显示图列
    plt.show()

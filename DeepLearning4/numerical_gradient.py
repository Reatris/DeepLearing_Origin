#求梯度函数
import numpy as np
import numerical_diff as nmf
def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x) #生成和x形状相同的数组
    

    #先计算单个维度的变量偏导再还原变量计算其他偏导
    for  idx in range(x.shape[0]):#如果这里可以是x.size x.shape[0&1] 看情况定
        temp_val=x[idx]
        #f(x+h)的计算
        x[idx]=temp_val+h
        fxh1=f(x)

        #f(x-h)计算
        x[idx]=temp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=temp_val#还原这个变量 
        
    return  grad

if __name__  == '__main__':
    print('现在求function_2（x[0]**2+x[1]**2）在 （3，4）处的偏导数')
    target=numerical_gradient(nmf.function_2,np.array([3.0,4.0]))
    print('结果为'+str(target))

#对softmax函数进行改进
#softmax用以非正则化的结果素组处理求取最大概率
import numpy as np
def softmax(a,batch_size):
    c=np.max(a)
    exp_a=np.exp(a-c)#正确无误
    sum_exp_a=np.sum(exp_a)#就是这个地方计入batch_size倍
    y=(exp_a/sum_exp_a)*batch_size 
    return y#返回的batch_size的每个元组之和为100%

if __name__ =='__main__':
    print('现在对softmax函数进行测试')
    print("现在有np数组[998.0,999.0,997.5,666.6,444.4]")
    test=np.array([998.0,999.0,997.5,666.6,444.4])
    target=softmax(test)
    print('正则化结果是'+str(target))

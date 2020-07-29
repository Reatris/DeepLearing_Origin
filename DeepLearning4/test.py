import numpy as np
'''
x_acc=np.arange(0,1201,600)#x_acc=[0,600,1200]
print(x_acc)
'''
'''
a = []
text=np.array([[1,2,3,4],[1,2,2,2]])
text2=np.array([[1,1,2,2],[2,3,3,2]])
print(text)
print(text2)
print(text.shape[0])
print(text.shape[1])

a.append(np.sum(text2 == text))
print(np.sum(text2 == text))#作用是返回两个数组中对应元素相等的个数

print(a)
'''
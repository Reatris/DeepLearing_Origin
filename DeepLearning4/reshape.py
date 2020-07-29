#测试结果reshape不改变原来的数组，只能新复值
import numpy as np
list=[1,2,3,4,5,6]
a=np.array([1,2,3,4,5,8])
print(list)
arr=np.array(list)
print(arr)
arr.reshape(2,3)
print(arr)
d=a.reshape((2,3))
print(d)

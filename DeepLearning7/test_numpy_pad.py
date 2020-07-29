import numpy as np
a=np.array([[1,1],[1,1]])
print(a)
b=np.pad(a,((1,1),(1,1)),'constant',constant_values=(0,9))
print(b)
#结论 二维先是竖着填充在是横着填充
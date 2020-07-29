import im2col as Im2col
import numpy as np

x1=np.random.rand(1,3,7,7)
print(x1.shape)
col1=Im2col.im2col(x1,5,5,stride=1,pad=0)
print('这是x1变换之后的样子')
print(col1.shape)

x2=np.random.rand(10,3,7,7)
print(x2.shape)
col2=Im2col.im2col(x2,5,5,stride=1,pad=0)
print('这是x2变换之后的样子')
print(col2.shape)

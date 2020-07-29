import col2im as Col2im
import im2col as Im2col
import numpy as np

test=np.random.randn(4,4,4,4)
print(test[1,2,3])
print(test.shape)
im2col_test=Im2col.im2col(test,5,5,1,0)
print(im2col_test.shape)
#col2im_test=Col2im.col2im(im2col_test,im2col_test.shape,5,5,1,0)
import numerical_diff as mdf
import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0.0,20,0.1) #范围0--20 单位0.1
y=mdf.funation_1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.show()

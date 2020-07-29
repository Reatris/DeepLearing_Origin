#求导函数
def mumerical_diff(f,x):
    h=1e-4   #0.0001
    return (f(x+h)-f(x-h))/(2*h)
#这是要求导的目标
def function_1(x):
    return 2*(x**2)
#多元函数（求偏导数）
def function_2(x):#这里x是一个numpy数组(设定2元)
    return x[0]**2+x[1]**2

if __name__ == '__main__':
    print('求函数f(x)=2*(x**2)在x=2处的倒数')
    result=mumerical_diff(function_1,2)
    print('结果是'+str(result))
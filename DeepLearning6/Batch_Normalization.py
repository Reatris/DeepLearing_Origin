class Batch_Normalization:
    def __init__(self,x,gamma,beta,bn_param):
        """
  输入:
  - x: 输入数据 shape (N, D)
  - gamma: 缩放参数 shape (D,)
  - beta: 平移参数 shape (D,)
  - bn_param: 包含如下参数的dict:
    - mode: 'train' or 'test'; 用来区分训练还是测试
    - eps: 除以方差时为了防止方差太小而导致数值计算不稳定
    - momentum: 前面讨论的momentum.
    - running_mean: 数组 shape (D,) 记录最新的均值
    - running_var 数组 shape (D,) 记录最新的方差
  返回一个tuple:
  - out: shape (N, D)
  - cache: 缓存反向计算时需要的变量
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
 
  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
 
  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    x_mean=x.mean(axis=0)
    x_var=x.var(axis=0)
    x_normalized=(x-x_mean)/np.sqrt(x_var+eps)
    out = gamma * x_normalized + beta
 
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
    cache = (x, x_mean, x_var, x_normalized, beta, gamma, eps)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_normalized = (x - running_mean)/np.sqrt(running_var +eps)
    out = gamma*x_normalized + beta
 
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
 
  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
 
  return out, cache
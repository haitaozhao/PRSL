import numpy as np
# sigmoid function
def my_sigmoid(w,x):
    return  1/(1+np.exp(-w.T.dot(x.T)))
# 损失函数
def obj_fun(w,x,y):
    tmp = y.reshape(1,-1)*np.log(my_sigmoid(w,x)) + \
    (1-y.reshape(1,-1))*np.log(1-my_sigmoid(w,x))
    return np.sum(-tmp)
# 计算随机梯度的函数
def my_Stgrad(w,x,y):
    return (my_sigmoid(w,x) - y)*x.T
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
mat = scipy.io.loadmat('.//data_Perc.mat')
Data = mat['x']
label = mat['y']
X = np.c_[np.ones([15,1]),Data.T]

w = np.zeros([3,1])

# 计算预测结果
result = X.dot(w).reshape(-1)
pred = 1*(result>0)
print(pred)
# label属于{-1,1},转成{0,1},方便后续编程
nlabel = (label+1)/2
print(nlabel)
pred != nlabel

# 梯度下降求解w，r为学习率，t为迭代次数
r = 0.01
t = 0
error = []
while np.sum(pred != nlabel) and t<2000:
    # 分类错误的样本和标签
    WX = X[np.array((pred != nlabel)).reshape(-1),:]
    Wlabel = label[0,np.array((pred != nlabel)).reshape(-1)]
    # 梯度的负方向
    deltaw =  WX.T.dot(Wlabel.reshape(-1,1))
    wt = w + r* deltaw
    # 目标函数值
    error.append(-(WX.dot(w)/(np.linalg.norm(w)+np.spacing(1))).T.dot(label[0,np.array((pred != nlabel)).reshape(-1)].T))
    t= t+1
    pred = []
    w = wt
    result = X.dot(w).reshape(-1)
    pred = 1*(result>0)
print("the number of iteration is %d" %t)   

# 画图
ax1 = plt.subplot(212)
ax1.plot(error[1:],label='objective function')
ax1.legend()

ax2 = plt.subplot(221) 
a = (label.reshape(-1)>0).nonzero()
a = np.array(a)
Dp = Data[:,a.reshape(-1)]
ax2.plot(Data[0,:],Data[1,:],'.')
ax2.plot(Dp[0,:],Dp[1,:],'r+')
ax2.set_title('Training Data')

ax3 = plt.subplot(222)
ax3.plot(Data[0,:],Data[1,:],'.')
ax3.plot(Dp[0,:],Dp[1,:],'r+')

p1 = 0
p2 = (-wt[1]*p1-wt[0])/wt[2]
q1 = 12
q2 = (-wt[1]*q1-wt[0])/wt[2]
ax3.plot([p1,q1],[p2,q2],'k-')
ax3.set_title('Decision boundary of Perceptron')
plt.show()


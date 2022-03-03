import numpy as np

def my_nb(x,gnd):
    a,b = x.shape
    my_lab = np.unique(gnd)
    NumOfClass = my_lab.size
    pw = np.zeros([NumOfClass])
    my_m = np.zeros([a,NumOfClass,1])
    my_std = np.zeros([a,NumOfClass,1])
    for i in range(NumOfClass):
        temp = np.sum(gnd==my_lab[i])
        pw[i] = temp/gnd.size
    
    for i in range(a):
        for j in range(NumOfClass):
            tmpX = x[i][np.where(gnd==my_lab[j])]
            my_m[i][j][0] = np.mean(tmpX)
            my_std[i][j][0] = np.std(tmpX)
            tmpX = []

    return pw,my_m,my_std




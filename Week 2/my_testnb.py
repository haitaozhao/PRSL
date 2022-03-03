from pdb import post_mortem
import numpy as np
import scipy.stats as st
def my_testnb(xt,pw,my_mean,my_std,NumOfClass,NumVar):
    a,b = xt.shape
    post_p = np.zeros([b,NumOfClass])
    test_lab = np.zeros(b)
    for k in range(b):
        temp = xt[:,k]
        for i in range(NumOfClass):
            prod = 1
            for j in range(NumVar):
                prod = prod*st.norm.pdf(temp[j],my_mean[j][i],my_std[j][i])

            post_p[k][i] = prod*pw[i]

        test_lab[k] = post_p[k].argmax()
    return post_p,test_lab


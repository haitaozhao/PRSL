from my_nb import *
from my_testnb import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

pw,my_m,my_std = my_nb(X_train.T,y_train)
post_p,test_lab = my_testnb(
    X_test.T,pw,my_m,my_std,3,4)
right = y_test == test_lab
rate = np.sum(right)/y_test.size 
print('The accuracy is %f' % rate)



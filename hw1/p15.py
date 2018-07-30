import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from plot import *

train_data = np.loadtxt('features.train')
x_train, y_train = train_data[:,1:], train_data[:,0] == 0
test_data = np.loadtxt('features.test')
x_test, y_test = test_data[:,1:], test_data[:,0] == 0

X0, X1 = x_train[:, 0], x_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

for logg in [0, 1, 2, 3, 4]:
    svm = SVC(
            C=0.1,
            kernel='rbf',
            gamma=10**logg)
    svm.fit(x_train, y_train)
    
    print('log gamma:', logg)
    print('Eout: {:.4f}'.format(1 - svm.score(x_test, y_test)))
    continue
    plot_contours(plt, svm, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=0.1)
    plt.show()



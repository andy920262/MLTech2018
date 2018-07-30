import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('features.train')
x_train, y_train = train_data[:,1:], train_data[:,0] == 8

ein, nsv = [], []

for logc in [-5, -3, -1, 1, 3]:
    svm = SVC(
            C=10**logc,
            kernel='poly',
            degree=2,
            gamma=1,
            coef0=1)
    svm.fit(x_train, y_train)
    print('logC:', logc)
    print('Ein: {:.4f}'.format(1 - svm.score(x_train, y_train)))
    print('# SV:', svm.n_support_.sum())
    ein.append(1 - svm.score(x_train, y_train))
    nsv.append(svm.n_support_.sum())

plt.plot([-5, -3, -1, 1, 3], ein)
plt.xlabel('log C')
plt.ylabel('Ein')
plt.show()

plt.plot([-5, -3, -1, 1, 3], nsv)
plt.xlabel('log C')
plt.ylabel('# SV')
plt.show()


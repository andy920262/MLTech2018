import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('features.train')
x_train, y_train = train_data[:,1:], train_data[:,0] == 0

w = []

for logc in [-5, -3, -1, 1, 3]:
    svm = SVC(
            C=10**logc,
            kernel='linear')
    svm.fit(x_train, y_train)
    print('logC:', logc)
    print('|w|: {:.4f}'.format(np.sqrt(np.dot(svm.coef_, svm.coef_.T))[0, 0]))
    
    w.append(np.sqrt(np.dot(svm.coef_, svm.coef_.T))[0, 0])

plt.plot([-5, -3, -1, 1, 3], w)
plt.xlabel('log C')
plt.ylabel('|w|')
plt.show()


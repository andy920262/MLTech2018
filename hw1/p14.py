import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('features.train')
x_train, y_train = train_data[:,1:], train_data[:,0] == 0

dist = []

for logc in [-3, -2, -1, 0, 1]:
    svm = SVC(
            C=10**logc,
            kernel='rbf',
            gamma=80)
    svm.fit(x_train, y_train)
    # w = \alpha * \phi(x_n)
    # w^2 = \alpha^2 * \phi(x_n)^2
    # w^2 = \alpha^2 * K(x_n, x_n)
    # |w| = sqrt(\alpha^2 * K(x_n, x_n))
    w = 0
    for sv in svm.support_vectors_:
        w += np.dot(svm.dual_coef_**2, np.exp(-80 * np.sqrt(np.sum((svm.support_vectors_ - sv)**2, -1))))
    print('logC:', logc)
    print('dist: {:.4f}'.format(1 / np.sqrt(w)[0]))
    dist.append(1 / np.sqrt(w)[0])

plt.plot([-3, -2, -1, 0, 1], dist)
plt.xlabel('log C')
plt.ylabel('distance')
plt.show()



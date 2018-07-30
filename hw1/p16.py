import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

train_data = np.loadtxt('features.train')
x_data, y_data = train_data[:,1:], train_data[:,0] == 0

gamma = []
for t in range(100):
    best = 0
    idx = np.arange(x_data.shape[0])
    np.random.shuffle(idx)
    x_train, y_train = x_data[idx[1000:]], y_data[idx[1000:]]
    x_test, y_test = x_data[idx[:1000]], y_data[idx[:1000]]
    for logg in [-1, 0, 1, 2, 3]:
        svm = SVC(
                C=0.1,
                kernel='rbf',
                gamma=10**logg)
        svm.fit(x_train, y_train)
        
        acc = svm.score(x_test, y_test)
        if acc > best:
            best = acc
            ret = logg
    gamma.append(ret)
plt.hist(gamma)
plt.xticks(range(-1,4))
plt.xlabel('log gamma')
plt.show()


import numpy as np
from sklearn.svm import SVC

def kernel(x1, x2):
    return (1 + 2 * np.dot(x1.transpose, x2))**2

x_data = np.array([(1,0),(0,1),(0,-1),(-1,0),(0,2),(0,-2),(-2,0)])
y_data = np.array([-1,-1,-1,1,1,1,1])

svm = SVC(
        kernel='poly',
        degree=2,
        gamma=2,
        coef0=1)
svm.fit(x_data, y_data)
print('SV:', svm.support_vectors_)
print('alpha:', svm.dual_coef_)
b = y_data[svm.support_] - np.sum(svm.dual_coef_[0] * (1 + 2 * np.dot(svm.support_vectors_, svm.support_vectors_.T))**2, -1)
w = (np.vstack([svm.dual_coef_[0]] * 5).T * np.hstack([4 * svm.support_vectors_**2, 4 * svm.support_vectors_, np.ones((5,1))])).sum(0)
print(np.dot(np.hstack([x_data**2, x_data, np.ones((7, 1))]), w.T) + b.mean())
#print((svm.dual_coef_[0] * (1 + 2 * np.dot(x_data, svm.support_vectors_.T))**2).sum(-1) + b.mean())
#print(svm.decision_function(x_data))
w[-1] += b.mean()
print('{:.2f}*x1^2 + {:.2f}*x2^2 + {:.2f}*x1 + {:.2f}*x2 + {:.2}'.format(*w))

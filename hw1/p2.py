import numpy as np
from sklearn.svm import SVC

x_data = np.array([(1,0),(0,1),(0,-1),(-1,0),(0,2),(0,-2),(-2,0)])
y_data = np.array([-1,-1,-1,1,1,1,1])

svm = SVC(
        kernel='poly',
        degree=2,
        gamma=2,
        coef0=1)
svm.fit(x_data, y_data)
print(svm.support_)
print(svm.dual_coef_ * y_data[svm.support_])


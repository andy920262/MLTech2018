import numpy as np
from sklearn.kernel_ridge import KernelRidge
from IPython import embed

data = np.loadtxt('hw2_lssvm_all.dat')
train_data = data[:400]
test_data = data[400:]
x_train, y_train = train_data[:,:-1], train_data[:,-1]
x_test, y_test = test_data[:,:-1], test_data[:,-1]

for gamma in [32, 2, 0.125]:
    for lamb in [0.001, 1, 1000]:
        krr = KernelRidge(
                alpha=lamb,
                kernel='rbf',
                gamma=gamma)
        krr.fit(x_train, y_train)
        ein = (np.sign(krr.predict(x_train)) != y_train).mean()
        eout = (np.sign(krr.predict(x_test)) != y_test).mean()
        print('gamma={} lambda={} ein={} eout={}'.format(gamma, lamb, ein, eout))

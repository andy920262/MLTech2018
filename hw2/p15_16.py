import numpy as np
from sklearn.kernel_ridge import KernelRidge
from IPython import embed

data = np.loadtxt('hw2_lssvm_all.dat')
data = np.hstack([np.ones((500, 1)), data])
train_data = data[:400]
test_data = data[400:]
x_train, y_train = train_data[:,:-1], train_data[:,-1]
x_test, y_test = test_data[:,:-1], test_data[:,-1]

for lamb in [0.01, 0.1, 1, 10, 100]:
    ein = np.zeros(400)
    eout = np.zeros(100)
    for i in range(250):
        lrr = KernelRidge(alpha=lamb)
        idx = np.random.choice(400, size=400)
        lrr.fit(x_train[idx], y_train[idx])
        ein += np.sign(lrr.predict(x_train))
        eout += np.sign(lrr.predict(x_test))
    ein = (np.sign(ein) != y_train).mean()
    eout = (np.sign(eout) != y_test).mean()
    print('lambda={} ein={} eout={}'.format(lamb, ein, eout))

import numpy as np
import matplotlib.pyplot as plt

def trans(x):
    return (2*x[1]**2-4*x[0]+2, x[0]**2-2*x[1]-1)

x_data = [(1,0),(0,1),(0,-1),(-1,0),(0,2),(0,-2),(-2,0)]
y_data = [-1,-1,-1,1,1,1,1]
z_pos = [trans(x) for x, y in zip(x_data,y_data) if y == 1]
z_neg = [trans(x) for x, y in zip(x_data,y_data) if y == -1]

plt.plot(*zip(*z_pos), 'go', label='z(y=+1)')
plt.plot(*zip(*z_neg), 'rx', label='z(y=-1)')
plt.plot([5]*50, np.linspace(-10, 10), label='z1=5')
plt.xlabel('z1')
plt.ylabel('z2')
plt.axis([-5,15,-10,10])
plt.legend()
plt.show()

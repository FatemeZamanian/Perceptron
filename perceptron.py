import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data=pd.read_csv('linear_data_train.csv')
X1=data['X1']
X2=data[' X2']
X1=np.array(X1)
X2=np.array(X2)
X1=X1.reshape(1000,1)
X2=X2.reshape(1000,1)
X=np.concatenate((X1,X2),axis=1)
Y=data[' Y']
Y=np.array(Y)
Y=Y.reshape(1000,1)

w=np.random.rand(2,1)
lr=0.001

x_range = np.arange(X[:,0].min(), X[:,0].max())
y_range = np.arange(X[:,1].min(), X[:,1].max())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(1000):
    y_pred=np.matmul(X[i,:],w)
    e=Y[i]-y_pred
    w= w + lr * X[i, :].T * e
    x, y = np.meshgrid(x_range, y_range)
    z = x * w[0] + y * w[1]

    ax.clear()
    ax.plot_surface(x, y, z, alpha = 0.4)
    ax.scatter(X[:,0],X[:,1],Y, c='blue')
    ax.set_xlabel('X0')
    ax.set_ylabel('X1')
    ax.set_zlabel('Y')
    plt.pause(0.001)
    plt.show()

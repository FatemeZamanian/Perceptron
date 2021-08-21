from os import error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#train_dataset
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

#perceptron
def fit(X,Y):
    w=np.random.rand(2)
    lr=0.001
    x_range = np.arange(X[:,0].min(), X[:,0].max(),0.01)
    y_range = np.arange(X[:,1].min(), X[:,1].max(),0.01)
    x_range=x_range[0:100]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(-140, 60)
    errors=[]
    for i in range(X.shape[0]):
        y_pred=np.matmul(X[i],w)
        e=Y[i]-y_pred

        Y_pred = np.matmul(X, w)
        Y_pred=Y_pred.reshape(1000,1)
        error = np.mean(np.abs(Y- Y_pred))
        errors.append(error)

        w= w + lr * X[i, :]* e
        x, y = np.meshgrid(x_range, y_range)
        z = w[0]*x + w[1]*y
        #show train data
        # ax.clear()
        # ax.plot_surface(x, y, z, rstride=1, cstride=1,  alpha = 0.4)
        # ax.scatter(X[:,0],X[:,1],Y, c='blue')
        # ax.set_xlabel('X0')
        # ax.set_ylabel('X1')
        # ax.set_zlabel('Y')
        # plt.pause(0.01)
    #plt.show()
    return errors,w

errors,w=fit(X,Y)
print(w)
# show error plot 
errors=np.array(errors)
# x = np.arange(0,1000,1)
# print(errors)
# y = np.array(errors)
# plt.plot(x, y)
# plt.show()


def predict(X_test):
    Y_test=[]
    for i in range(X_test.shape[0]):
        y_pred=np.matmul(X_test[i],w)
        if y_pred>0:
            Y_test.append(1)
        else:
            Y_test.append(-1)
    return Y_test

def evaluate_accuracy(X,Y):
    y_predict = predict(X)
    count=0
    for i in range(len(Y)):
        if y_predict[i]==Y[i]:
            count+=1
    accuracy=count/len(Y)
    return accuracy

def evaluate_error(X,Y):
    y_predict = predict(X)
    error = np.mean(np.abs(Y - y_predict))
    return error
    


#train_dataset
test=pd.read_csv('linear_data_test.csv')
Xt1=test['X1']
Xt2=test[' X2']
Xt1=np.array(Xt1)
Xt2=np.array(Xt2)
Xt1=Xt1.reshape(200,1)
Xt2=Xt2.reshape(200,1)
X_t=np.concatenate((Xt1,Xt2),axis=1)
Yt=test[' Y']
Yt=np.array(Yt)
Y_t=Yt.reshape(200,1)
y_pr=predict(X_t)
acc_test=evaluate_accuracy(X_t,Y_t)
acc_train=evaluate_accuracy(X,Y)
error_test=evaluate_error(X_t,Y_t)
error_train=evaluate_error(X,Y)
print('test accuracy :',acc_test)
print('train accuracy :',acc_train)
print('test error :',error_test)
print('train error :',error_train)
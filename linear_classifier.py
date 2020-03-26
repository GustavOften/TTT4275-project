import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import platform
import itertools

if(platform.system() == 'Windows'): #Windows
    class_1 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_1', delimiter=',')
    class_2 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_2', delimiter=',')
    class_3 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_3', delimiter=',')
else:  #Linux
    class_1 = genfromtxt('task_1/Iris_TTT4275/class_1', delimiter=',')
    class_2 = genfromtxt('task_1/Iris_TTT4275/class_2', delimiter=',')
    class_3 = genfromtxt('task_1/Iris_TTT4275/class_3', delimiter=',')
features = 4
W = np.ones((3,features+1))*0.01
tk_1 = np.zeros((3,1))
tk_1[0,0] = 1
tk_2 = np.zeros((3,1))
tk_2[1,0] = 1
tk_3 = np.zeros((3,1))
tk_3[2,0] = 1
norm_grad_MSE = 100
alpha = 0.00001
for i in range(1000):
    N=40
    grad_MSE = np.zeros((3,features+1))
    for n in range(N):
        for (_class,tk) in zip([class_1,class_2,class_3],[tk_1,tk_2,tk_3]):
            xk = np.append(_class[n,:],1).reshape((features+1,1))
            gk = W@xk
            temp = np.multiply(gk-tk,gk)
            temp = np.multiply(temp, np.ones((3,1))-gk)
            grad_MSE += alpha*temp@xk.T
    #print(f'grad_MSE:{grad_MSE}')
    W -= grad_MSE
    norm_grad_MSE = np.linalg.norm(grad_MSE)
    #print(W)
np.savetxt('w.txt',W)
W = np.loadtxt('w.txt')

###############################
### Classification ################
confusion = np.zeros([3,3])
for n in range(50):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        g = W@np.append(class_[n,:],1).reshape(features+1,1)
        print(g.reshape(3))
        classified_class = np.argmax(g, axis=0)
        confusion[i,classified_class] += 1
        i += 1

print(confusion)
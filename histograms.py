import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt

class_1 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_1', delimiter=',')
class_2 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_2', delimiter=',')
class_3 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_3', delimiter=',')


plt.hist(class_1[:,0])
plt.hist(class_2[:,0])
plt.hist(class_3[:,0])
plt.show()
plt.hist(class_1[:,1])
plt.hist(class_2[:,1])
plt.hist(class_3[:,1])
plt.show()
plt.hist(class_1[:,2])
plt.hist(class_2[:,2])
plt.hist(class_3[:,2])
plt.show()
plt.hist(class_1[:,3])
plt.hist(class_2[:,3])
plt.hist(class_3[:,3])
plt.show()
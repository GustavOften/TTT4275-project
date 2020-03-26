import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt
import platform

if(platform.system() == 'Windows'): #Windows
    class_1 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_1', delimiter=',')
    class_2 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_2', delimiter=',')
    class_3 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_3', delimiter=',')
else:  #Linux
    class_1 = genfromtxt('task_1/Iris_TTT4275/class_1', delimiter=',')
    class_2 = genfromtxt('task_1/Iris_TTT4275/class_2', delimiter=',')
    class_3 = genfromtxt('task_1/Iris_TTT4275/class_3', delimiter=',')




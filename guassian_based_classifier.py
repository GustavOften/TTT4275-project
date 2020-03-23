import numpy as np 
from numpy import genfromtxt
import matplotlib.pyplot as plt

gaussian = lambda sigma, mu, x: np.sqrt(2*np.pi)**len(-x/2)*np.linalg.det(sigma)**(-1/2)\
    *np.exp(-1/2*(mu-x)@np.linalg.inv(sigma)@(mu-x))

class_1 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_1', delimiter=',')
class_2 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_2', delimiter=',')
class_3 = genfromtxt('..\TTT4275-project\\task_1\Iris_TTT4275\class_3', delimiter=',')
mu_1 = sum(class_1[0:30,:])/30
mu_2 = sum(class_2[0:30,:])/30
mu_3 = sum(class_3[0:30,:])/30
sigma_1 = (class_1[0:30,:]-mu_1).T@(class_1[0:30,:]-mu_1)/30
sigma_2 = (class_2[0:30,:]-mu_2).T@(class_2[0:30,:]-mu_2)/30
sigma_3 = (class_3[0:30,:]-mu_3).T@(class_3[0:30,:]-mu_3)/30
print(sigma_1)
print(sigma_2)
print(sigma_3)
confusion_training_data = np.zeros([3,3])
for n in range(30):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        guassians = [gaussian(sigma_1,mu_1, class_[n,:]),\
                    gaussian(sigma_2,mu_2, class_[n,:]),\
                    gaussian(sigma_3,mu_3, class_[n,:])]
        classified_class = guassians.index(max(guassians))
        confusion_training_data[i,classified_class] += 1
        i += 1
print("First 30 samples for training, last 20 for testing:")
print(" ")
print("The confusion matrix for the training data, 30 samples for training:")
print(confusion_training_data)
confusion_testing_data = np.zeros([3,3])
for n in range(20):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        guassians = [gaussian(sigma_1,mu_1, class_[n+30,:]),\
                    gaussian(sigma_2,mu_2, class_[n+30,:]),\
                    gaussian(sigma_3,mu_3, class_[n+30,:])]
        classified_class = guassians.index(max(guassians))
        confusion_testing_data[i,classified_class] += 1
        i += 1
print("The confusion matrix for the testing data, 30 samples for training:")
print(confusion_testing_data)
#############################################
#############################################
#############################################
mu_1 = sum(class_1[20:50,:])/30
mu_2 = sum(class_2[20:50,:])/30
mu_3 = sum(class_3[20:50,:])/30
sigma_1 = (class_1[20:50,:]-mu_1).T@(class_1[20:50,:]-mu_1)/30
sigma_2 = (class_2[20:50,:]-mu_2).T@(class_2[20:50,:]-mu_2)/30
sigma_3 = (class_3[20:50,:]-mu_3).T@(class_3[20:50,:]-mu_3)/30

confusion_training_data = np.zeros([3,3])
for n in range(30):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        guassians = [gaussian(sigma_1,mu_1, class_[n+20,:]),\
                    gaussian(sigma_2,mu_2, class_[n+20,:]),\
                    gaussian(sigma_3,mu_3, class_[n+20,:])]
        classified_class = guassians.index(max(guassians))
        confusion_training_data[i,classified_class] += 1
        i += 1
print("Last 30 samples for training, first 20 for testing:")
print(" ")
print("The confusion matrix for the training data, 30 samples for training:")
print(confusion_training_data)
confusion_testing_data = np.zeros([3,3])
for n in range(20):
    i = 0
    for class_ in [class_1, class_2, class_3]:
        guassians = [gaussian(sigma_1,mu_1, class_[n,:]),\
                    gaussian(sigma_2,mu_2, class_[n,:]),\
                    gaussian(sigma_3,mu_3, class_[n,:])]
        classified_class = guassians.index(max(guassians))
        confusion_testing_data[i,classified_class] += 1
        i += 1
print("The confusion matrix for the testing data, 30 samples for training:")
print(confusion_testing_data)


#plt.hist(class_1[:,0])
#plt.hist(class_2[:,0])
#plt.hist(class_3[:,0])
#plt.show()
#plt.hist(class_1[:,1])
#plt.hist(class_2[:,1])
#plt.hist(class_3[:,1])
#plt.show()
#plt.hist(class_1[:,2])
#plt.hist(class_2[:,2])
#plt.hist(class_3[:,2])
#plt.show()
#plt.hist(class_1[:,3])
#plt.hist(class_2[:,3])
#plt.hist(class_3[:,3])
#plt.show()

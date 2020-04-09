import numpy as np
import pandas as pd
from numpy import genfromtxt
import platform
import matplotlib.pyplot as plt

gaussian = lambda sigma, mu, x: np.sqrt(2*np.pi)**len(-x/2)*np.linalg.det(sigma)**(-1/2)\
    *np.exp(-1/2*(mu-x)@np.linalg.inv(sigma)@(mu-x))

if(platform.system() == 'Windows'): #Windows
    data = pd.read_csv('..\TTT4275-project\\task_2\Wovels\\vowdata.dat', sep="\s+", header=24, index_col=0)
    #data = data.transpose()
else:  #Linux
    data = pd.read_csv('task_2/Wovels/vowdata.dat', sep="\s+", header=24, index_col=0)
    data = data.transpose()


### Structure of data: 
#character 1:     m=man, w=woman, b=boy, g=girl
#characters 2-3:  talker number
#characters 4-5:  vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
#                        ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
#                        oo="hood", uh="hud", uw="who'd")
gender = np.array(['m','w','b','g'])
range_m = [0, 44]; range_f = [45,92]; range_b = [93, 119]; range_g = [120,138]
train_r_m = [0,22]; train_r_f = [45,67]; train_r_b = [93, 106]; train_r_g = [120, 129]
test_r_m = [23, 44]; test_r_f = [68, 92]; test_r_b = [107, 119]; test_r_g = [130, 138]

vowels = np.array(['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih','iy','oa','oo','uh','uw'])

numpy_array_data = data.to_numpy()
ae = numpy_array_data[0:139]
ah = numpy_array_data[139:278]
aw = numpy_array_data[278:417]
eh = numpy_array_data[417:556]
er = numpy_array_data[556:695]
ei = numpy_array_data[695:834]
ih = numpy_array_data[834:973]
iy = numpy_array_data[973:1112]
oa = numpy_array_data[1112:1251]
oo = numpy_array_data[1251:1390]
uh = numpy_array_data[1390:1529]
uw = numpy_array_data[1529:1668]

ae_test = np.concatenate((ae[train_r_m[0]:train_r_m[1], :],\
                        ae[train_r_f[0]:train_r_f[1], :],\
                        ae[train_r_b[0]:train_r_b[1], :],\
                        ae[train_r_g[0]:train_r_g[1], :]), axis=0)

# Sums the elements and divides by the number of elements that are not zero.
mu_ae = sum(ae_test)/np.count_nonzero(ae_test, axis=0)
# Creats a vector with the number of nonzero features 
non_zero = np.matrix(np.count_nonzero(ae_test, axis=0))
# Finds the position of every element that is zero
zero_mask = (ae_test == 0)
# Replaces the value of zero with the mean value, this will give an added zero for sigma
ae_test[zero_mask] = mu_ae[np.where(zero_mask)[1]]
# Creats a matrix with the number of features used for calculating each covariance
number_of_elements = np.minimum(np.full((15,15), non_zero), np.full((15,15),non_zero.T))
# Calculates the covariance
sigma_ae = (ae_test - mu_ae).T@(ae_test-mu_ae)/number_of_elements



##mu_aw = sum(class_2[0:30,:])/30
#mu_eh = sum(class_3[0:30,:])/30
#sigma_1 = (class_1[0:30,:]-mu_1).T@(class_1[0:30,:]-mu_1)/30
#sigma_2 = (class_2[0:30,:]-mu_2).T@(class_2[0:30,:]-mu_2)/30

#for i in range(15):
#    plt.hist(ae[range_m[0]:range_m[1],i], label='man')
#    plt.hist(ae[range_f[0]:range_f[1],i], label='woman')
#    plt.hist(ae[range_b[0]:range_b[1],i], label='boy')
#    plt.hist(ae[range_g[0]:range_g[1],i], label='girl')
#    plt.axvline(x=mu_ae[i], label='mu')
#    plt.legend()
#    plt.show()





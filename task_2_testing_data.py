import numpy as np
import pandas as pd
from numpy import genfromtxt
import platform
import matplotlib.pyplot as plt

def get_mu_and_full_covariance_matrix(data):
    # Sums the elements and divides by the number of elements that are not zero.
    mu = sum(data)/np.count_nonzero(data, axis=0)
    # Creats a vector with the number of nonzero features 
    non_zero = np.matrix(np.count_nonzero(data, axis=0))
    # Finds the position of every element that is zero
    zero_mask = (data == 0)
    # Replaces the value of zero with the mean value, this will give an added zero for sigma
    data[zero_mask] = mu[np.where(zero_mask)[1]]
    # Creats a matrix with the number of features used for calculating each covariance
    number_of_elements = np.minimum(np.full((15,15), non_zero), np.full((15,15),non_zero.T))
    # Calculates the covariance
    sigma = (data - mu).T@(data-mu)/number_of_elements
    return mu, sigma

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

ae_train = np.concatenate((ae[train_r_m[0]:train_r_m[1], :],\
                        ae[train_r_f[0]:train_r_f[1], :],\
                        ae[train_r_b[0]:train_r_b[1], :],\
                        ae[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_ae, sigma_ae = get_mu_and_full_covariance_matrix(ae_train)

ah_train = np.concatenate((ah[train_r_m[0]:train_r_m[1], :],\
                        ah[train_r_f[0]:train_r_f[1], :],\
                        ah[train_r_b[0]:train_r_b[1], :],\
                        ah[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_ah, sigma_ah = get_mu_and_full_covariance_matrix(ah_train)

aw_train = np.concatenate((aw[train_r_m[0]:train_r_m[1], :],\
                        aw[train_r_f[0]:train_r_f[1], :],\
                        aw[train_r_b[0]:train_r_b[1], :],\
                        aw[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_aw, sigma_aw = get_mu_and_full_covariance_matrix(aw_train)

eh_train = np.concatenate((eh[train_r_m[0]:train_r_m[1], :],\
                        eh[train_r_f[0]:train_r_f[1], :],\
                        eh[train_r_b[0]:train_r_b[1], :],\
                        eh[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_eh, sigma_eh = get_mu_and_full_covariance_matrix(eh_train)

er_train = np.concatenate((er[train_r_m[0]:train_r_m[1], :],\
                        er[train_r_f[0]:train_r_f[1], :],\
                        er[train_r_b[0]:train_r_b[1], :],\
                        er[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_er, sigma_er = get_mu_and_full_covariance_matrix(er_train)

ei_train = np.concatenate((ei[train_r_m[0]:train_r_m[1], :],\
                        ei[train_r_f[0]:train_r_f[1], :],\
                        ei[train_r_b[0]:train_r_b[1], :],\
                        ei[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_ei, sigma_ei = get_mu_and_full_covariance_matrix(ei_train)

ih_train = np.concatenate((ih[train_r_m[0]:train_r_m[1], :],\
                        ih[train_r_f[0]:train_r_f[1], :],\
                        ih[train_r_b[0]:train_r_b[1], :],\
                        ih[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_ih, sigma_ih = get_mu_and_full_covariance_matrix(ih_train)

iy_train = np.concatenate((iy[train_r_m[0]:train_r_m[1], :],\
                        iy[train_r_f[0]:train_r_f[1], :],\
                        iy[train_r_b[0]:train_r_b[1], :],\
                        iy[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_iy, sigma_iy = get_mu_and_full_covariance_matrix(iy_train)

oa_train = np.concatenate((oa[train_r_m[0]:train_r_m[1], :],\
                        oa[train_r_f[0]:train_r_f[1], :],\
                        oa[train_r_b[0]:train_r_b[1], :],\
                        oa[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_oa, sigma_oa = get_mu_and_full_covariance_matrix(oa_train)

oo_train = np.concatenate((oo[train_r_m[0]:train_r_m[1], :],\
                        oo[train_r_f[0]:train_r_f[1], :],\
                        oo[train_r_b[0]:train_r_b[1], :],\
                        oo[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_oo, sigma_oo = get_mu_and_full_covariance_matrix(oo_train)

uh_train = np.concatenate((uh[train_r_m[0]:train_r_m[1], :],\
                        uh[train_r_f[0]:train_r_f[1], :],\
                        uh[train_r_b[0]:train_r_b[1], :],\
                        uh[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_uh, sigma_uh = get_mu_and_full_covariance_matrix(uh_train)

uw_train = np.concatenate((uw[train_r_m[0]:train_r_m[1], :],\
                        uw[train_r_f[0]:train_r_f[1], :],\
                        uw[train_r_b[0]:train_r_b[1], :],\
                        uw[train_r_g[0]:train_r_g[1], :]), axis=0)
mu_uw, sigma_uw = get_mu_and_full_covariance_matrix(uw_train)

#for i in range(15):
#    plt.hist(ae[range_m[0]:range_m[1],i], label='man')
#    plt.hist(ae[range_f[0]:range_f[1],i], label='woman')
#    plt.hist(ae[range_b[0]:range_b[1],i], label='boy')
#    plt.hist(ae[range_g[0]:range_g[1],i], label='girl')
#    plt.axvline(x=mu_ae[i], label='mu')
#    plt.legend()
#    plt.show()





import numpy as np
import pandas as pd
from numpy import genfromtxt
import platform
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from data_behandling import replace_zeros_with_mean
def gmm(sigma,mu,weights,x):
    p = 0
    for i in range(0,len(weights)):
        p += weights[i]*gaussian(np.diag(sigma[i]),mu[i],x) #change np.diag if cov_typ is full
        #p += weights[i]*gaussian(sigma[i],mu[i],x)
    return p


def find_error_rate(confusion):
    N=np.sum(confusion)
    error_counter=0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[0]):
            if(i != j):
                error_counter += confusion[i,j]
    return(error_counter/N)

gaussian = lambda sigma, mu, x: np.sqrt(2*np.pi)**len(-x/2)*np.linalg.det(sigma)**(-1/2)\
    *np.exp(-1/2*(mu-x)@np.linalg.inv(sigma)@(mu-x))

if(platform.system() == 'Windows'): #Windows
    data = pd.read_csv('..\TTT4275-project\\task_2\Wovels\\vowdata.dat', sep="\s+", header=24, index_col=0)
    #data = data.transpose()
else:  #Linux
    data = pd.read_csv('task_2/Wovels/vowdata.dat', sep="\s+", header=24, index_col=0)
    #data = data.transpose()



### Structure of data: 
#character 1:     m=man, w=woman, b=boy, g=girl
#characters 2-3:  talker number
#characters 4-5:  vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
#                        ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
#                        oo="hood", uh="hud", uw="who'd")


range_m = [0, 44]; range_f = [45,92]; range_b = [93, 119]; range_g = [120,138]
train_r_m = [0,22]; train_r_f = [45,67]; train_r_b = [93, 106]; train_r_g = [120, 129]
test_r_m = [23, 44]; test_r_f = [68, 92]; test_r_b = [107, 119]; test_r_g = [130, 138]

numpy_array_data = data.to_numpy()
numpy_array_data = replace_zeros_with_mean(numpy_array_data)


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

ah_train = np.concatenate((ah[train_r_m[0]:train_r_m[1], :],\
                        ah[train_r_f[0]:train_r_f[1], :],\
                        ah[train_r_b[0]:train_r_b[1], :],\
                        ah[train_r_g[0]:train_r_g[1], :]), axis=0)

aw_train = np.concatenate((aw[train_r_m[0]:train_r_m[1], :],\
                        aw[train_r_f[0]:train_r_f[1], :],\
                        aw[train_r_b[0]:train_r_b[1], :],\
                        aw[train_r_g[0]:train_r_g[1], :]), axis=0)

eh_train = np.concatenate((eh[train_r_m[0]:train_r_m[1], :],\
                        eh[train_r_f[0]:train_r_f[1], :],\
                        eh[train_r_b[0]:train_r_b[1], :],\
                        eh[train_r_g[0]:train_r_g[1], :]), axis=0)

er_train = np.concatenate((er[train_r_m[0]:train_r_m[1], :],\
                        er[train_r_f[0]:train_r_f[1], :],\
                        er[train_r_b[0]:train_r_b[1], :],\
                        er[train_r_g[0]:train_r_g[1], :]), axis=0)

ei_train = np.concatenate((ei[train_r_m[0]:train_r_m[1], :],\
                        ei[train_r_f[0]:train_r_f[1], :],\
                        ei[train_r_b[0]:train_r_b[1], :],\
                        ei[train_r_g[0]:train_r_g[1], :]), axis=0)

ih_train = np.concatenate((ih[train_r_m[0]:train_r_m[1], :],\
                        ih[train_r_f[0]:train_r_f[1], :],\
                        ih[train_r_b[0]:train_r_b[1], :],\
                        ih[train_r_g[0]:train_r_g[1], :]), axis=0)

iy_train = np.concatenate((iy[train_r_m[0]:train_r_m[1], :],\
                        iy[train_r_f[0]:train_r_f[1], :],\
                        iy[train_r_b[0]:train_r_b[1], :],\
                        iy[train_r_g[0]:train_r_g[1], :]), axis=0)

oa_train = np.concatenate((oa[train_r_m[0]:train_r_m[1], :],\
                        oa[train_r_f[0]:train_r_f[1], :],\
                        oa[train_r_b[0]:train_r_b[1], :],\
                        oa[train_r_g[0]:train_r_g[1], :]), axis=0)

oo_train = np.concatenate((oo[train_r_m[0]:train_r_m[1], :],\
                        oo[train_r_f[0]:train_r_f[1], :],\
                        oo[train_r_b[0]:train_r_b[1], :],\
                        oo[train_r_g[0]:train_r_g[1], :]), axis=0)

uh_train = np.concatenate((uh[train_r_m[0]:train_r_m[1], :],\
                        uh[train_r_f[0]:train_r_f[1], :],\
                        uh[train_r_b[0]:train_r_b[1], :],\
                        uh[train_r_g[0]:train_r_g[1], :]), axis=0)

uw_train = np.concatenate((uw[train_r_m[0]:train_r_m[1], :],\
                        uw[train_r_f[0]:train_r_f[1], :],\
                        uw[train_r_b[0]:train_r_b[1], :],\
                        uw[train_r_g[0]:train_r_g[1], :]), axis=0)

ae_test = np.concatenate((ae[test_r_m[0]:test_r_m[1], :],\
                        ae[test_r_f[0]:test_r_f[1], :],\
                        ae[test_r_b[0]:test_r_b[1], :],\
                        ae[test_r_g[0]:test_r_g[1], :]), axis=0) 

ah_test = np.concatenate((ah[test_r_m[0]:test_r_m[1], :],\
                        ah[test_r_f[0]:test_r_f[1], :],\
                        ah[test_r_b[0]:test_r_b[1], :],\
                        ah[test_r_g[0]:test_r_g[1], :]), axis=0)

aw_test = np.concatenate((aw[test_r_m[0]:test_r_m[1], :],\
                        aw[test_r_f[0]:test_r_f[1], :],\
                        aw[test_r_b[0]:test_r_b[1], :],\
                        aw[test_r_g[0]:test_r_g[1], :]), axis=0)

eh_test = np.concatenate((eh[test_r_m[0]:test_r_m[1], :],\
                        eh[test_r_f[0]:test_r_f[1], :],\
                        eh[test_r_b[0]:test_r_b[1], :],\
                        eh[test_r_g[0]:test_r_g[1], :]), axis=0)

er_test = np.concatenate((er[test_r_m[0]:test_r_m[1], :],\
                        er[test_r_f[0]:test_r_f[1], :],\
                        er[test_r_b[0]:test_r_b[1], :],\
                        er[test_r_g[0]:test_r_g[1], :]), axis=0)             

ei_test = np.concatenate((ei[test_r_m[0]:test_r_m[1], :],\
                        ei[test_r_f[0]:test_r_f[1], :],\
                        ei[test_r_b[0]:test_r_b[1], :],\
                        ei[test_r_g[0]:test_r_g[1], :]), axis=0)

ih_test = np.concatenate((ih[test_r_m[0]:test_r_m[1], :],\
                        ih[test_r_f[0]:test_r_f[1], :],\
                        ih[test_r_b[0]:test_r_b[1], :],\
                        ih[test_r_g[0]:test_r_g[1], :]), axis=0)

iy_test = np.concatenate((iy[test_r_m[0]:test_r_m[1], :],\
                        iy[test_r_f[0]:test_r_f[1], :],\
                        iy[test_r_b[0]:test_r_b[1], :],\
                        iy[test_r_g[0]:test_r_g[1], :]), axis=0)

oa_test = np.concatenate((oa[test_r_m[0]:test_r_m[1], :],\
                        oa[test_r_f[0]:test_r_f[1], :],\
                        oa[test_r_b[0]:test_r_b[1], :],\
                        oa[test_r_g[0]:test_r_g[1], :]), axis=0)

oo_test = np.concatenate((oo[test_r_m[0]:test_r_m[1], :],\
                        oo[test_r_f[0]:test_r_f[1], :],\
                        oo[test_r_b[0]:test_r_b[1], :],\
                        oo[test_r_g[0]:test_r_g[1], :]), axis=0)

uh_test = np.concatenate((uh[test_r_m[0]:test_r_m[1], :],\
                        uh[test_r_f[0]:test_r_f[1], :],\
                        uh[test_r_b[0]:test_r_b[1], :],\
                        uh[test_r_g[0]:test_r_g[1], :]), axis=0)

uw_test = np.concatenate((uw[test_r_m[0]:test_r_m[1], :],\
                        uw[test_r_f[0]:test_r_f[1], :],\
                        uw[test_r_b[0]:test_r_b[1], :],\
                        uw[test_r_g[0]:test_r_g[1], :]), axis=0)

confusion = np.zeros([15,15])
classes_train = np.array([ae_train, ah_train, aw_train, eh_train, er_train, ei_train, ih_train, iy_train, oa_train, oo_train, uh_train, uw_train])
classes = np.array([ae_test, ah_test, aw_test, eh_test, er_test, ei_test, ih_test, iy_test, oa_test, oo_test, uh_test, uw_test])
clas_labels = np.array(['ae','ah','aw','eh','er','ei','ih','iy','oa','oo','uh','uw'])

############################
######## Create model########

n_gaussians=3
cov_typ='diag' #'diag' or 'full'

classifyers=dict()
for (lbl,data_train) in zip(clas_labels,classes_train):
    classifyers[lbl]=GaussianMixture(n_gaussians,covariance_type=cov_typ,max_iter=500,tol=0.00001) #create model
    classifyers[lbl].fit(data_train) #train model




############################
######## Classify ########
confusion = np.zeros([12,12])
for n in range(65):
    k = 0
    for class_ in classes:
        temp = []
        for i in clas_labels:
            sigma=classifyers[i].covariances_
            mu=classifyers[i].means_
            weights=classifyers[i].weights_ 
            temp.append(gmm(sigma,mu,weights,class_[n,:]))            
        classified_class = temp.index(max(temp))
        confusion[k, classified_class] += 1
        k += 1
print(confusion)


print(f'Error rate for diagonal covariance:{find_error_rate(confusion)}')

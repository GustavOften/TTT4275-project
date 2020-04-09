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

for i in range(15):
    plt.hist(uh[range_m[0]:range_m[1],i], label='man')
    plt.hist(uh[range_f[0]:range_f[1],i], label='woman')
    plt.hist(uh[range_b[0]:range_b[1],i], label='boy')
    plt.hist(uh[range_g[0]:range_g[1],i], label='girl')
    plt.legend()
    plt.show()




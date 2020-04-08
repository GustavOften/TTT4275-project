import numpy as np
import pandas as pd
from numpy import genfromtxt
import platform


if(platform.system() == 'Windows'): #Windows
    data = pd.read_csv('..\TTT4275-project\\task_2\Wovels\\vowdata.dat', sep="\s+", header=24, index_col=0)
    data = data.transpose()
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
range_for_genders = np.array([50, 50, 29, 29])
vowels = np.array(['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih','iy','oa','oo','uh','uw'])
for i in range(range_for_genders[0]):
    print(data[gender[0]+f"{i+1:02d}"+vowels[0]])
    
import numpy as np
import pandas as pd
from numpy import genfromtxt
import platform

if(platform.system() == 'Windows'): #Windows
    data = pd.read_csv('..\TTT4275-project\\task_2\Wovels\\vowdata.dat', sep="\s+", header=24)
else:  #Linux
    data = pd.read_csv('task_2/Wovels/vowdata.dat', sep="\s+", header=24)




print(data['1'])

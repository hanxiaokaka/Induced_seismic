import numpy as np
from peakdetector import pk_indxs
import matplotlib.pyplot as plt

vector = np.array([2,10,2,3,1,3,6,4,6,5,6,6,6,6,2,1,9,0,8,8,6,5,6,7,8,9,0,0,0])
print('Detect peaks with minimum height and distance filters.')
indexes = pk_indxs(vector,trshd=0.2, min_dist=2)
print('Peaks are: %s' % (indexes))
print(type(indexes))

fig = plt.figure()
plt.plot(np.arange(0,len(vector)),vector,color='#FFDD44')
plt.plot(indexes,vector[indexes],'o',color='r')
plt.show()



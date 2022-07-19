import numpy as np
from peakdetector import pk_indxs
import matplotlib.pyplot as plt
from orion_light import seismic_catalog
from scipy.interpolate import interp1d

fname = '/Users/pmh/Desktop/cushingSeismic.hdf5'
catalog = seismic_catalog.SeismicCatalog()
catalog.load_catalog_hdf5(fname)
event_times = catalog.get_epoch_slice()
new_t=np.linspace(0,event_times[-1] - event_times[0],num=1000)/86400/365.25
n_cnts = np.arange(len(event_times))
t_n = (event_times - event_times[0])/86400/365.25
n_func = interp1d(t_n, n_cnts, kind='linear')
new_n=n_func(new_t)
seis_rate=np.diff(new_n)

indexes = pk_indxs(seis_rate,trshd=0.05, min_dist=50)

fig, axs = plt.subplots(2)
axs[0].plot(new_t[1:],seis_rate,color='#FFDD44')
axs[0].plot(new_t[indexes],seis_rate[indexes],'o',color='r')
axs[0].set_title('Seismic Rate')
axs[0].set(xlabel='', ylabel='Number')

axs[1].plot(new_t,new_n,color='r')
axs[1].set_title('Accumulated Number')
axs[1].set(xlabel='Time (Years)', ylabel='Number')

plt.show()



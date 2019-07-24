from scipy.io.idl import readsav
import numpy as np
import matplotlib.pyplot as plt
from helita.sim import rh15d
import h5py
import xarray

dpath_atmos = '/mn/stornext/u3/souvikb/rh/Atmos/'

atmos = xarray.open_dataset(dpath_atmos+'FALC_82_5x5.hdf5')

z = np.asarray(atmos['z'])
T = np.asarray(atmos['temperature'])
vz =np.asarray(atmos['velocity_z'])
nH =np.asarray(atmos['hydrogen_populations'])
vturb = np.asarray(atmos['velocity_turbulent'])
ne = np.asarray(atmos['electron_density'])

vz[0,0,1,:40]=22000.
vz[0,0,1,40:]=0.

rh15d.make_xarray_atmos('/mn/stornext/u3/souvikb/rh/Atmos/FALC_vel_try.hdf5',T=T,z=z,nH=nH,vz =vz,vturb=vturb,ne=ne,desc='FALC_Modified-atmos_with step velocity at one pixel')

#So far the best result was achieved with vz from 0 to 17 km/s in a step function. The saved atmospehere 
# was /mn/stornext/u3/souvikb/rh/Atmos/FALC_vel1.hdf5
# FALC_vel_try.hdf5 was even better than FALC_vel1.hdf5




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

#vz[0,0,0,:35]=10000
#vz[0,0,0,35:]=0.
print(vturb[0,0,0,:])  # added 8 km/s based on the k2v and k2r peak difference between cluster no. 21 in Cak and FALC   
vturb[0,0,0,:]=vturb[0,0,0,:]*0#8000. # synthesized profiles . It turned out 8 km/s would be needed to match the "width" of
print(vturb[0,0,0,:]) #synthetics with observations.

rh15d.make_xarray_atmos('/mn/stornext/u3/souvikb/rh/Atmos/FALC_mic.hdf5',T=T,z=z,nH=nH,vz =vz,vturb=vturb,ne=ne,desc='FALC_Modified-atmos_with an added microturbulence of 8 km/s.')





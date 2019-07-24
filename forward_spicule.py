import numpy as np
import astropy
import matplotlib.pyplot as plt
from helita.sim import rh15d
from astropy.io import fits
import xarray

#-----------------Atmosphere-----------------#

#atmos_r = xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_82_5x5.hdf5')

atmos_20= xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_20.hdf5')
atmos_25 =xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_25.hdf5')
atmos_30 =xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_30.hdf5')
 
vz_20 =np.asarray(atmos_20['velocity_z']) # (0,1) is the pixel with proper velocity; use (1,1) for rest i.e. 0
vz_25 =np.asarray(atmos_25['velocity_z'])
vz_30 =np.asarray(atmos_30['velocity_z'])

z = np.asarray(atmos_20['z'])

atmos_neg20= xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_-20.hdf5')
atmos_neg25= xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_-25.hdf5')
atmos_neg30= xarray.open_dataset('/mn/stornext/u3/souvikb/rh/Atmos/FALC_new_interp_-30.hdf5')

vz_neg20 =np.asarray(atmos_neg20['velocity_z']) # (0,1) is the pixel with proper velocity; use (1,1) for rest i.e. 0
vz_neg25 =np.asarray(atmos_neg25['velocity_z'])
vz_neg30 =np.asarray(atmos_neg30['velocity_z'])


#My code here####

################## FOR Ca K ########################

dataCa20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_20/') #Reading Output
waveCa = dataCa20.ray.wavelength # reads the entire frequency spectrum for the active atom
indices_Ca = np.arange(len(waveCa))[(waveCa > 393.234) & (waveCa < 393.5)]

I_Ca20 = dataCa20.ray.intensity # Specific intensity from the Radiative transfer equation

dataCa25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_25/')
dataCa30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_30/')
I_Ca25 = dataCa25.ray.intensity
I_Ca30 = dataCa30.ray.intensity

dataCa_r = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_rest/')

dataCaneg_20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_-20/')
dataCaneg_25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_-25/')
dataCaneg_30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Ca/output/output_-30/')

I_Caneg20 = dataCaneg_20.ray.intensity
I_Caneg25 = dataCaneg_25.ray.intensity
I_Caneg30 = dataCaneg_30.ray.intensity

I_Ca_r = dataCa_r.ray.intensity

################## FOR Mg k #######################


dataMg20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_20/')
dataMg25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_25/')
dataMg30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_30/')
waveMg = dataMg20.ray.wavelength
indices_Mg = np.arange(len(waveMg))[(waveMg >279.52 ) & (waveMg < 279.78)]

I_Mg20 = dataMg20.ray.intensity
I_Mg25 = dataMg25.ray.intensity
I_Mg30 = dataMg30.ray.intensity

dataMg_r = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output_rest/')

I_Mg_r = dataMg_r.ray.intensity

dataMgneg_20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_-20/')
dataMgneg_25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_-25/')
dataMgneg_30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_Mg/output/output_-30/')

I_Mgneg20 = dataMgneg_20.ray.intensity
I_Mgneg25 = dataMgneg_25.ray.intensity
I_Mgneg30 = dataMgneg_30.ray.intensity

################# FOR H ##########################

dataH20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_20/')
dataH25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_25/')
dataH30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_30/')
waveH = dataH20.ray.wavelength
indices_H = np.arange(len(waveH))[(waveH >656.1) & (waveH < 656.5)]

I_H20 = dataH20.ray.intensity
I_H25 = dataH25.ray.intensity
I_H30 = dataH30.ray.intensity

dataH_r= rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_rest/')

I_H_r = dataH_r.ray.intensity

dataHneg_20 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_-20/')
dataHneg_25 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_-25/')
dataHneg_30 = rh15d.Rh15dout('/mn/stornext/u3/souvikb/rh/rh15d/run_H/output/output_-30/')

I_Hneg20 = dataHneg_20.ray.intensity
I_Hneg25 = dataHneg_25.ray.intensity
I_Hneg30 = dataHneg_30.ray.intensity


#################Doppler axes####################

dopp_H = ((waveH[indices_H]-656.28)/656.28)*3e5
dopp_Ca = ((waveCa[indices_Ca]-393.367)/393.367)*3e5
dopp_Mg = ((waveMg[indices_Mg]-279.633)/279.633)*3e5
##################Plotting routine ###############


fig, axs =plt.subplots(2,4,figsize=(12,6),facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.18,wspace=0.33,left=0.076,right=0.974,top=0.809,bottom=0.191)
axs=axs.ravel()

axs[0].plot(dopp_H,I_H_r[0,0,indices_H],color='black',linestyle='dashed')
axs[0].set_title(r'H $\alpha$')
#axs[0].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[0].set_ylabel('I [$W m^{-2} Hz^{-1} Sr^{-1}$]')
axs[0].plot(dopp_H,I_H20[0,0,indices_H],color='black')
axs[0].plot(dopp_H,I_H25[0,0,indices_H],color='red')
axs[0].plot(dopp_H,I_H30[0,0,indices_H],color='blue')
axs[0].set_xlim([dopp_H[0],dopp_H[52]])

axs[1].plot(dopp_Ca,I_Ca_r[0,0,indices_Ca],color='black',linestyle='dashed')
axs[1].set_title(r'Ca II K')
#axs[1].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
#axs[0].set_ylabel('I [$W m^{-2} Hz^{-1} Sr^{-1}$]')
axs[1].plot(dopp_Ca,I_Ca20[0,0,indices_Ca],color='black')
axs[1].plot(dopp_Ca,I_Ca25[0,0,indices_Ca],color='red')
axs[1].plot(dopp_Ca,I_Ca30[0,0,indices_Ca],color='blue')
axs[1].set_xlim([dopp_H[0],dopp_H[52]])

axs[2].plot(dopp_Mg,I_Mg_r[0,0,indices_Mg],color='black',linestyle='dashed')
axs[2].set_title(r'Mg II k')
#axs[2].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[2].plot(dopp_Mg,I_Mg20[0,0,indices_Mg],color='black')
axs[2].plot(dopp_Mg,I_Mg25[0,0,indices_Mg],color='red')
axs[2].plot(dopp_Mg,I_Mg30[0,0,indices_Mg],color='blue')
axs[2].set_xlim([dopp_H[0],dopp_H[52]])

axs[3].plot(z[0,:]/1e6,-vz_20[0,1,1,:]/1e3,color='black',linestyle='dashed')
axs[3].plot(z[0,:]/1e6,-vz_20[0,0,1,:]/1e3,color='black')
axs[3].plot(z[0,:]/1e6,-vz_25[0,0,1,:]/1e3,color='red')
axs[3].plot(z[0,:]/1e6,-vz_30[0,0,1,:]/1e3,color='blue')
axs[3].set_title('LOS velocity')
#axs[3].set_xlabel('Height[Mm]')
axs[3].set_ylabel('V$_{z}$ [km s$^{-1}$]')

axs[4].plot(dopp_H,I_H_r[0,0,indices_H],color='black',linestyle='dashed')
#axs[4].set_title(r'H $\alpha$')
axs[4].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[4].set_ylabel('I [$W m^{-2} Hz^{-1} Sr^{-1}$]')
axs[4].plot(dopp_H,I_Hneg20[0,0,indices_H],color='black')
axs[4].plot(dopp_H,I_Hneg25[0,0,indices_H],color='red')
axs[4].plot(dopp_H,I_Hneg30[0,0,indices_H],color='blue')
axs[4].set_xlim([dopp_H[0],dopp_H[52]])

axs[5].plot(dopp_Ca,I_Ca_r[0,0,indices_Ca],color='black',linestyle='dashed')
#axs[5].set_title(r'Ca II K')
axs[5].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
#axs[0].set_ylabel('I [$W m^{-2} Hz^{-1} Sr^{-1}$]')
axs[5].plot(dopp_Ca,I_Caneg20[0,0,indices_Ca],color='black')
axs[5].plot(dopp_Ca,I_Caneg25[0,0,indices_Ca],color='red')
axs[5].plot(dopp_Ca,I_Caneg30[0,0,indices_Ca],color='blue')
axs[5].set_xlim([dopp_H[0],dopp_H[52]])

axs[6].plot(dopp_Mg,I_Mg_r[0,0,indices_Mg],color='black',linestyle='dashed')
#axs[6].set_title(r'Mg II k')
axs[6].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[6].plot(dopp_Mg,I_Mgneg20[0,0,indices_Mg],color='black')
axs[6].plot(dopp_Mg,I_Mgneg25[0,0,indices_Mg],color='red')
axs[6].plot(dopp_Mg,I_Mgneg30[0,0,indices_Mg],color='blue')
axs[6].set_xlim([dopp_H[0],dopp_H[52]])

axs[7].plot(z[0,:]/1e6,-vz_20[0,1,1,:]/1e3,color='black',linestyle='dashed')
axs[7].plot(z[0,:]/1e6,-vz_neg20[0,0,1,:]/1e3,color='black')
axs[7].plot(z[0,:]/1e6,-vz_neg25[0,0,1,:]/1e3,color='red')
axs[7].plot(z[0,:]/1e6,-vz_neg30[0,0,1,:]/1e3,color='blue')
#axs[7].set_title('LOS velocity gradient')
axs[7].set_xlabel('Height[Mm]')
axs[7].set_ylabel('V$_{z}$ [km s$^{-1}$]')

print(dopp_H)
plt.show()
#plt.tight_layout()

#top=0.9,
#bottom=0.191,
#left=0.076,
#right=0.974,
#hspace=0.17,
#wspace=0.3

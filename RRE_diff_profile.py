import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA
from astropy.io import fits
import sys
from helita.io import lp
import pickle
from scipy.io.idl import readsav
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib import colors

dpath='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'
dpath_kmean_bose = '/mn/stornext/d9/souvikb/K_means_results/savefiles/'

hdrCa_im = lp.getheader(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris.fcube')
hdrCa_sp = lp.getheader(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris_sp.fcube')
hdrH_im =lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')
hdrH_sp = lp.getheader(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris_sp.icube')

dimCa_im = hdrCa_im[0]
dimCa_sp = hdrCa_sp[0]
dimH_im = hdrH_im[0]
dimH_sp = hdrH_sp[0]

cubeCa = lp.getdata(dpath+'crispex_3950_2017-05-25T09:12:00_scans=0-424_time-corrected_rotated2iris.fcube')
cubeH = lp.getdata(dpath+'crispex_6563_08:05:00_aligned_3950_2017-05-25T09:12:00_scans=0-424_rotated2iris.icube')

cubeCa = np.reshape(cubeCa,[dimCa_im[0],dimCa_im[1],dimCa_sp[1],dimCa_sp[0]])
cubeH = np.reshape(cubeH,[dimH_im[0],dimH_im[1],dimH_sp[1],dimH_sp[0]])

cube_Mg = lp.getdata(dpath+'IRIS_Mg_SJI_2796_07:33:00_intblown_aligned_3950_2017-05-25T09:12:00_rotated.fcube')

scan = ((154,155,164,165,167,168,194,195,196,197))
m1a = np.mean(cubeCa[400:699,600:899,scan,41])
m2a = np.mean(cubeH[400:699,600:899,scan,31])

cubeCa1 = cubeCa[:,:,scan,:]/m1a
cubeH1 = cubeH[:,:,scan,:]/m2a

meanCa = np.mean(cubeCa1[:,:,:,[0,40]],axis=3)
#mean2 = np.mean(cube2[:,:,[0,30]],axis=2)
meanH = cubeH1[:,:,:,31]

m1b = np.median(cubeCa1[:,:,:,[0,40]])
m2b = np.median(cubeH1[:,:,:,[0,30]])

cubeCa1 = cubeCa1[:,:,:,:]/meanCa[:,:,:,None]*m1b
cubeH1 = cubeH1[:,:,:,:]/meanH[:,:,:,None]

dpath_kmean ='/mn/stornext/d9/souvikb/K_means_results/'

labels=pickle.load(open(dpath_kmean+'kmeans_labels.pickle','rb'))
data = pickle.load(open(dpath_kmean+'kmeans_training.pickle','rb')) #(50,72)
wave_H= readsav(dpath+'spectfile.6563.idlsave')
wave_Ca =readsav(dpath+'spectfile.3950.idlsave')
wave_H=wave_H['spect_pos']
wave_Ca=wave_Ca['spect_pos']
H_avg = np.mean(np.mean(cubeH[400:699,600:899,6,:],axis=0),axis=0)

dopp_H = ((wave_H-6563.)/6563.)*3e5
dopp_Ca = ((wave_Ca[:41]-3933.7)/3933.7)*3e5

prof_H = np.zeros((10,6,32)) # This is where the profiles will be stored.
prof_Ca = np.zeros((10,6,41))# This is where the profiles of Ca will be stored.

clusts = [18,46,8,26,36,40]

for j in range(10):
 for i in range(6):
   mask_H = cubeH[:,:,scan[j],:]*0
   res = np.where(labels[:,:,j]==clusts[i])
   mask_H[res[0],res[1],:]=1
   pdt_H = cubeH[:,:,scan[j],:]*mask_H
   prof_H[j,i,:] = np.mean(pdt_H[res[0],res[1],:],axis=0)

   mask_Ca = cubeCa[:,:,scan[j],:41]*0
   mask_Ca[res[0],res[1],:]=1
   pdt_Ca = cubeCa[:,:,scan[j],:41]*mask_Ca
   prof_Ca[j,i,:]= np.mean(pdt_Ca[res[0],res[1],:],axis=0)

result_prof_H = np.mean(prof_H,axis=0)
result_prof_Ca = np.mean(prof_Ca,axis=0)

mask1 = np.zeros((1518,1641,5))

wav= readsav(dpath_kmean_bose+'IRIS_wavs.sav')
wav_mg= wav['wav_mg']
dopp_Mg = ((wav_mg[40:75]-2796.35)/2796.35)*3e5

clust_Mg_8 = readsav(dpath_kmean_bose+'cluster_8_Mg.sav')
clust_Mg_8 = clust_Mg_8['mg_clust8']

clust_Mg_26 = readsav(dpath_kmean_bose+'cluster_26_Mg.sav')
clust_Mg_26 = clust_Mg_26['mg_clust26']

clust_Mg_36 = readsav(dpath_kmean_bose+'cluster_36_Mg.sav')
clust_Mg_36 = clust_Mg_36['mg_clust36']

clust_Mg_avg = readsav(dpath_kmean_bose+'cluster_avg_Mg.sav')
clust_Mg_avg = clust_Mg_avg['mg_clust_avg']

cmap1= colors.ListedColormap(['magenta','lime','blue','red','orange'])

for i in range(5):
  res = np.where(labels[:,:,3]==clusts[i])
  mask1[res[0],res[1],i]=1

fig, axs =plt.subplots(2,3,figsize=(12,8),facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.18,wspace=0.23,left=0.05,right=0.95,top=0.93,bottom=0.07)
axs=axs.ravel()

axs[0].imshow(np.transpose(cubeH[:,:,scan[3],26]),cmap='Greys_r',extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].contour(np.transpose(mask1[:,:,0]),levels=1,colors=['magenta'],alpha=1,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].contour(np.transpose(mask1[:,:,1]),levels=1,colors=['lime'],alpha=0.8,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].contour(np.transpose(mask1[:,:,2]),levels=1,colors=['blue'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].contour(np.transpose(mask1[:,:,3]),levels=1,colors=['red'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].contour(np.transpose(mask1[:,:,4]),levels=1,colors=['orange'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[0].set_title(r'CRISP H $\alpha$ +41 km s$^{-1}$')
axs[0].set_ylabel('arcsec')

axs[1].imshow(np.transpose(cubeCa[:,:,scan[3],24]**0.3),cmap='Greys_r',extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].contour(np.transpose(mask1[:,:,0]),levels=1,colors=['magenta'],alpha=1,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].contour(np.transpose(mask1[:,:,1]),levels=1,colors=['lime'],alpha=0.8,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].contour(np.transpose(mask1[:,:,2]),levels=1,colors=['blue'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].contour(np.transpose(mask1[:,:,3]),levels=1,colors=['red'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].contour(np.transpose(mask1[:,:,4]),levels=1,colors=['orange'],alpha=0.3,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[1].set_title(r'CHROMIS Ca II K +19.8 km s$^{-1}$')
axs[1].set_ylabel('arcsec')

axs[2].imshow(np.transpose(cube_Mg[:,:,scan[3]]),cmap='Greys_r',extent=[0,56,0,61],origin='lower',aspect='auto')
axs[2].contour(np.transpose(mask1[:,:,0]),levels=1,colors=['magenta'],alpha=1,extent=[0,56,0,61],origin='lower')
axs[2].contour(np.transpose(mask1[:,:,1]),levels=1,colors=['lime'],alpha=0.8,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[2].contour(np.transpose(mask1[:,:,2]),levels=1,colors=['blue'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[2].contour(np.transpose(mask1[:,:,3]),levels=1,colors=['red'],alpha=0.2,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[2].contour(np.transpose(mask1[:,:,4]),levels=1,colors=['orange'],alpha=0.3,extent=[0,56,0,61],origin='lower',aspect='auto')
axs[2].set_title(r'IRIS SJI 2796')
axs[2].set_ylabel('arcsec')

axs[3].plot(dopp_H, result_prof_H[0,:]/1e3,color='magenta')
axs[3].plot(dopp_H, result_prof_H[1,:]/1e3,color='lime')
axs[3].plot(dopp_H, result_prof_H[2,:]/1e3,color='blue')
axs[3].plot(dopp_H, result_prof_H[3,:]/1e3,color='red')
axs[3].plot(dopp_H, result_prof_H[4,:]/1e3,color='orange')
axs[3].plot(dopp_H, result_prof_H[5,:]/1e3,color='black',linestyle='dashed')
axs[3].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[3].set_ylabel('I$_{counts}$[x1000]')
axs[3].set_title(r'H $\alpha$')
axs[3].plot(dopp_H,np.abs(result_prof_H[5,:]/1e3-result_prof_H[0,:]/1e3)/1.2, linestyle='dotted',color='magenta')
axs[3].axvline(x=39.2,color='gray',linestyle='-.',zorder=-1)

axs[4].plot(dopp_Ca, result_prof_Ca[0,:41],color='magenta')
axs[4].plot(dopp_Ca, result_prof_Ca[1,:41],color='lime')
axs[4].plot(dopp_Ca, result_prof_Ca[2,:41],color='blue')
axs[4].plot(dopp_Ca, result_prof_Ca[3,:41],color='red')
axs[4].plot(dopp_Ca, result_prof_Ca[4,:41],color='orange')
axs[4].plot(dopp_Ca, result_prof_Ca[5,:41],color='black',linestyle='dashed')
axs[4].set_xlim([dopp_H[0],dopp_H[31]])
axs[4].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[4].set_ylabel('I [$W m^{-2} Hz^{-1} Sr^{-1}$]')
axs[4].set_title(r'Ca II K')
axs[4].plot(dopp_H,np.abs(result_prof_H[5,:]/1e3-result_prof_H[0,:]/1e3)/2e9, linestyle='dotted',color='magenta')
axs[4].axvline(x=39.2,color='gray',linestyle='-.',zorder=-1)

axs[5].plot(dopp_Mg,clust_Mg_8[40:75]/1e2,color='blue')
axs[5].plot(dopp_Mg,clust_Mg_26[40:75]/1e2,color='red')
axs[5].plot(dopp_Mg,clust_Mg_36[40:75]/1e2,color='orange')
axs[5].plot(dopp_Mg,clust_Mg_avg[40:75]/1e2,color='black',linestyle='dashed')
axs[5].set_xlim([dopp_H[0],dopp_H[31]])
axs[5].set_ylabel('I$_{counts}$[x100]')
axs[5].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[5].set_title(r'Mg II k')

sm = plt.cm.ScalarMappable(cmap=cmap1)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs,fraction=0.015,pad=0.05)
n_clusters=5
tick_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
cbar.set_ticks(tick_locs/4.)
cbar.set_ticklabels([8,7,6,5,4])
cbar.ax.set_title("RP Index No.")

plt.show()


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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,mark_inset
import matplotlib.colors as colors

#MEDIUM_SIZE = 11
#plt.rc('font', size=MEDIUM_SIZE)
dpath='/mn/stornext/d11/lapalma/reduc/2017/2017-05-25/CHROMIS/crispex/09:12:00/'

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

#dpath_kmean = '/sanhome/bose/KMEAN_results/'
dpath_kmean ='/mn/stornext/d9/souvikb/K_means_results/'

labels=pickle.load(open(dpath_kmean+'kmeans_labels.pickle','rb'))
data = pickle.load(open(dpath_kmean+'kmeans_training.pickle','rb')) #(50,72)

wave_H= readsav(dpath+'spectfile.6563.idlsave')
wave_Ca =readsav(dpath+'spectfile.3950.idlsave')
wave_H=wave_H['spect_pos']
wave_Ca=wave_Ca['spect_pos']
H_avg = np.mean(np.mean(cubeH[400:699,600:899,6,:],axis=0),axis=0)
Ca_avg = np.mean(np.mean(cubeCa[400:699,600:899,6,:],axis=0),axis=0)

H_avg1 = np.mean(np.mean(cubeH1[400:699,600:899,6,:],axis=0),axis=0)
Ca_avg1 = np.mean(np.mean(cubeCa1[400:699,600:899,6,:],axis=0),axis=0)

x_lam_ticks=[-50,0,50]
y_im_ticks =[0,20,40,60]
x_im_ticks =[0,20,40]

dopp_H = ((wave_H[:]-6563)/6563)*3e5
dopp_Ca = ((wave_Ca[:41]-3933.7)/3933.7)*3e5
prof_Ca = np.zeros((10,41))

for j in range(10):
  mask_Ca = cubeCa[:,:,scan[j],:41]*0
  res = np.where(labels[:,:,j]==40)
  mask_Ca[res[0],res[1],:]=1
  pdt_Ca = cubeCa[:,:,scan[j],:41]*mask_Ca
  prof_Ca[j,:] = np.mean(pdt_Ca[res[0],res[1],:],axis=0)

Ca_avg_QS = np.mean(prof_Ca,axis=0)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

image = np.transpose(labels[:,:,3])
rbeim = np.full(image.shape,-1)
rreim = np.full(image.shape,-1)
restim = np.full(image.shape,-1)

rbe = np.array([25, 49, 12, 48,])
rre = np.array([38, 26, 8, 46, 18,])
rest = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 42, 42, 43, 44, 45, 47,])

rbe_cn = np.zeros(rbe.size,dtype='int32')
rre_cn = np.zeros(rre.size,dtype='int32')
rest_cn = np.zeros(rest.size,dtype='int32')

for ii in range(rbe.size):
    ss = (image[:,:] == rbe[ii])
    rbeim[ss] = ii
    rbe_cn[ii]=ii

for ii in range(rre.size):
    ss = (image[:,:] == rre[ii])
    rreim[ss] = ii+rbe.size
    rre_cn[ii]=ii+rbe.size

for ii in range(rest.size):
    ss = (image[:,:] == rest[ii])
    restim[ss] = ii+rbe.size+rre.size
    rest_cn[ii]=ii+rbe.size+rre.size

cmap = plt.get_cmap('Greys',rest.size)
cmap1 = truncate_colormap(cmap, 0.0, 0.6, rest.size)

cmap = plt.get_cmap('Blues',rbe.size)
cmap2 = truncate_colormap(cmap, 0.5, 1.0, rbe.size)

cmap = plt.get_cmap('Reds',rre.size)
cmap3 = truncate_colormap(cmap, 0.5, 1.0, rre.size)


mrbeim = np.ma.masked_where(rbeim < 0, rbeim)
mrreim = np.ma.masked_where(rreim < 0, rreim)
mrestim = np.ma.masked_where(restim < 0, restim)
 
#fig3 = plt.figure(figsize=(12,12))

#gs = fig3.add_gridspec(3, 3,wspace=0.3,hspace=0.3)
#f3_ax1 = fig3.add_subplot(gs[0, 0])
fig, axs =plt.subplots(3,3,figsize=(12,15),facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 0.2,wspace=0.23,left=0.08,right=0.92,top=0.95,bottom=0.05)
axs=axs.ravel()

axs[0].imshow(np.transpose(cubeH[:,:,scan[3],8]),cmap='Greys_r',extent=[0,56,0,61],origin='lower',aspect='auto') #200:1099,300:1199,scan[6]
axins=inset_axes(axs[0],width="30%",height="30%",loc=4)
axins.imshow(np.transpose(cubeH[745:845,266:366,scan[3],8]),cmap='Greys_r',extent=[27.5,31.26,9.8,13.5],origin='lower',aspect='auto')
axins.text(29.32,11.0,"+",ha="center",color='red',fontsize=14)
#x1, x2, y1, y2 = 745, 845, 266, 366
#axins.set_xlim(x1, x2)
#axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')
#mark_inset(axs[0],axins,loc1=2,loc2=4,fc="none",ec="0.5")
axs[0].set_title(r'CRISP H $\alpha$ -41 km s$^{-1}$')
axs[0].plot([27.565,27.565],[9.84,14.5],color='black',alpha=0.5)
axs[0].plot([27.565,31.3],[14.5,14.5],color='black',alpha=0.5)
axs[0].plot([31.3,31.3],[14.5,9.84],color='black',alpha=0.5)
axs[0].plot([31.3,27.565],[9.84,9.84],color='black',alpha=0.5)
axs[0].set_yticks(y_im_ticks)
axs[0].set_xticks(x_im_ticks)
axs[0].text(10,55,"25.05.2017",ha="center",color='red')
#axs[0].text(29.32,11.,"+",ha="center",color='red',fontsize=12)
#f3_ax1.set_xlabel('acrsec')
axs[0].set_ylabel('arcsec')
#######################################

axs[1].imshow(np.transpose(cubeCa[:,:,scan[3],15])**0.3,cmap='Greys_r',extent=[0,56,0,61],origin='lower',aspect='auto')
axins=inset_axes(axs[1],width="30%",height="30%",loc=4)
axins.imshow(np.transpose(cubeCa[745:845,266:366,scan[3],15]),cmap='Greys_r',extent=[27.5,31.26,9.8,13.5],origin='lower',aspect='auto')
axins.text(29.32,11.,"+",ha="center",color='red',fontsize=14)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[1].plot([27.565,27.565],[9.84,14.5],color='black',alpha=0.5)
axs[1].plot([27.565,31.3],[14.5,14.5],color='black',alpha=0.5)
axs[1].plot([31.3,31.3],[14.5,9.84],color='black',alpha=0.5)
axs[1].plot([31.3,27.565],[9.84,9.84],color='black',alpha=0.5)
axs[1].set_title(r'CHROMIS Ca K -24.56 km s$^{-1}$')
#axs[1].text(29.32,11.,"+",ha="center",color='red',fontsize=12)
axs[1].set_yticks(y_im_ticks)
axs[1].set_xticks(x_im_ticks)
axs[1].set_ylabel('arcsec')
#plt.setp(f3_ax2.get_yticklabels(), visible=False)
#f3_ax3 = fig3.add_subplot(gs[0, 2],sharey=f3_ax2)
##im1=axs[2].imshow(np.transpose(labels[:,:,3]),cmap='Spectral_r',extent=[0,56,0,61],origin='lower',aspect='auto')
##axs[2].text(28,71.5,"RP Index No.",ha="center")
im1 =axs[2].imshow(mrestim,origin='lower',cmap=cmap1,extent=[0,56,0,61],aspect='auto')
im2=axs[2].imshow(mrbeim,origin='lower',cmap=cmap2,extent=[0,56,0,61],aspect='auto')
im3=axs[2].imshow(mrreim,origin='lower',cmap=cmap3,extent=[0,56,0,61],aspect='auto')

axins=inset_axes(axs[2],width="25%",height="25%",loc=4)
axins.imshow((mrestim[266:366,745:845]),cmap=cmap1,extent=[27.5,31.26,9.8,13.5],origin='lower',aspect='auto')
axins.imshow((mrbeim[266:366,745:845]),cmap=cmap2,extent=[27.5,31.26,9.8,13.5],origin='lower',aspect='auto')
axins.imshow((mrreim[266:366,745:845]),cmap=cmap3,extent=[27.5,31.26,9.8,13.5],origin='lower',aspect='auto')
axins.text(29.32,11.,"+",ha="center",color='white',fontsize=14)
axins.set_xticklabels('')
axins.set_yticklabels('')
axs[2].plot([27.565,27.565],[9.84,14.5],color='black',alpha=1)
axs[2].plot([27.565,31.3],[14.5,14.5],color='black',alpha=1)
axs[2].plot([31.3,31.3],[14.5,9.84],color='black',alpha=1)
axs[2].plot([31.3,27.565],[9.84,9.84],color='black',alpha=1)
axs[2].set_yticks(y_im_ticks)
axs[2].set_xticks(x_im_ticks)
#axs[2].text(29.32,11.,"+",ha="center",color='white',fontsize=12)
#f3_ax3.set_xlabel('acrsec')
axs[2].set_ylabel('arcsec')
axs[2].set_title('Representative Profiles (RP)')
#plt.setp(f3_ax3.get_yticklabels(), visible=False)
axs[2].text(64.5,62.85,"RP Index No.",ha="center",fontsize=12)
#divider = make_axes_locatable(axs[2])
#cax1 = divider.append_axes("right", size="3%", pad=0.04)
axins_cb1 = inset_axes(axs[2],
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1., 0., 1, 1),
                   bbox_transform=axs[2].transAxes,
                   borderpad=0,
                   )

axins_cb2 = inset_axes(axs[2],
                       width="5%",  # width = 5% of parent_bbox width
                       height="45%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.16, 0., 1, 1),
                       bbox_transform=axs[2].transAxes,
                       borderpad=0,
                       )

axins_cb3 = inset_axes(axs[2],
                       width="5%",  # width = 5% of parent_bbox width
                       height="45%",  # height : 50%
                       loc='upper left',
                       bbox_to_anchor=(1.16, 0., 1, 1),
                       bbox_transform=axs[2].transAxes,
                       borderpad=0,
                       )

plt.colorbar(im1,cax=axins_cb1,ticks = np.arange(np.min(rest_cn),np.max(rest_cn)+1,3),orientation="vertical")
axins_cb1.xaxis.set_ticks_position("default")
#cax2 = divider.append_axes("right", size="3%", pad=0.04)
plt.colorbar(im2,cax=axins_cb2,ticks = np.arange(np.min(rbe_cn),np.max(rbe_cn)+1),orientation="vertical")
axins_cb2.xaxis.set_ticks_position("default")

#cax3 = divider.append_axes("right", size="3%", pad=0.04)
plt.colorbar(im3,cax=axins_cb3,ticks = np.arange(np.min(rre_cn),np.max(rre_cn)+1),orientation="vertical")
axins_cb3.xaxis.set_ticks_position("default")

axs[4].plot(dopp_H[:31],H_avg[:31]/1e3,color='black',linestyle='dashed')
axs[4].plot(dopp_H[:31],cubeH[793,303,165,:31]/1e3,color='red') #854,927,194
#axs[4].text(0,6.4,"+",ha="center",color='red')
axs[4].set_xticks(x_lam_ticks)
axs[4].set_ylabel('I$_{counts}$[x1000]')
axs[4].set_title(r'H $\alpha$ spectra')
#axs[4].text(0,0.9,"+",ha="center",color='red')
#################################

axs[3].plot(dopp_H[:31],data[40,41:],color='black',linestyle='dashed')
axs[3].plot(dopp_H[:31],data[48,41:],color='red')
#axs[3].text(0,0.9,"+",ha="center",color='red')
axs[3].set_ylabel('I/I$_{w}$')
axs[3].set_xticks(x_lam_ticks)
axs[3].set_title(r'H $\alpha$ RP:3')
########################################
#f3_ax6 = fig3.add_subplot(gs[1, 2])
axs[5].imshow(cubeH[793,303,156:174,:31]/np.max(cubeH[793,303,156:174,:31]),cmap='Greys_r',extent=[dopp_H[0],dopp_H[31],-123,123],aspect='auto')
axs[5].yaxis.tick_right()
axs[5].yaxis.set_label_position("right")
axs[5].set_ylabel(r'$\Delta$ t[s]')
axs[5].set_title(r'H $\alpha$ $\lambda$-t')
axs[5].plot([-84.5,-74.5],[1,1],color='red')
axs[5].set_xticks(x_lam_ticks)
#f3_ax5.text(450,-155,"RP Index No.",ha="center")
##################################
#f3_ax7 = fig3.add_subplot(gs[2, 0])
#axs[7]=plt.axes()
axs[7].plot(dopp_Ca,Ca_avg_QS,color='black',linestyle='dashed')
axs[7].plot(dopp_Ca,cubeCa[793,303,165,:41],color='red')
#f3_ax7.set_xticks(x_lam_ca_ticks)
axs[7].set_xlim([dopp_H[0],dopp_H[30]])
axs[7].set_ylabel('I [$\mathrm{W m^{-2} Hz^{-1} Sr^{-1}}$]')
#axs[7].text(0,4.1e-09,"+",ha="center",color='red')
axs[7].set_title(r'Ca II  K spectra')
axs[7].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[7].set_xticks(x_lam_ticks)
#f3_ax7.axis([3932.2,3935,0.5e-09,4.5e-09])
#f3_ax7.text(3933.6,4e-09,"+",ha="center",color='red')
#axs[7].set_title(r'Ca K spectra')
#f3_ax7.grid()
#f3_ax7.set_ylabel('I$_{SP}$')
#f3_ax7.set_xlabel(r'$\lambda$[$\AA$]')
####################################
#f3_ax8 = fig3.add_subplot(gs[2, 1])
#axs[6]=plt.axes()
axs[6].plot(dopp_Ca,data[40,:41],color='black',linestyle='dashed')
axs[6].plot(dopp_Ca,data[48,:41],color='red')
#f3_ax8.set_xticks(x_lam_ca_ticks)
axs[6].set_xlim([dopp_H[0],dopp_H[30]])
#axs[6].text(0,0.135,"+",ha="center",color='red')
axs[6].set_ylabel('I/I$_{w}$')
axs[6].set_title('Ca II K RP:3')
axs[6].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[6].set_xticks(x_lam_ticks)
####################################
#f3_ax5.text(450,-155,"RP Index No.",ha="center")
#f3_ax5.set_title('gs[-1, -2]')
#f3_ax9 = fig3.add_subplot(gs[2, 2])
axs[8].imshow(cubeCa[793,303,156:174,:40]/np.max(cubeCa[793,303,156:174,:40]),cmap='Greys_r',extent=[dopp_H[0],dopp_H[31],-123,123],aspect='auto')
axs[8].yaxis.tick_right()
axs[8].yaxis.set_label_position("right")
axs[8].set_ylabel(r'$\Delta$ t[s]')
axs[8].set_title(r'Ca II K $\lambda$-t')
axs[8].set_xlabel(r'$\Delta$ $\lambda$[km s$^{-1}$]')
axs[8].plot([-84.5,-74.5],[1,1],color='red')
axs[8].set_xticks(x_lam_ticks)
#plt.tight_layout()


plt.show()

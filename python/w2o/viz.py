# Plotting utils

import numpy as np
import mne
import os
import matplotlib.pyplot as plt

from w2o import filesystem
from w2o import utils
from w2o import preliminary

import matplotlib as mpl
import matplotlib.pyplot as plt

########### FUNCTIONS

def get_vlims(data, real=False):
    
    ldata = data.copy().flatten()
    
    if real:
        vlims = (np.nanmin(ldata),np.nanmax(ldata))    
    else:
        if np.abs(np.sum(ldata/np.abs(ldata))) == ldata.shape[0]:
            if np.sum(ldata/np.abs(ldata)) > 0:            
                vlims = (np.nanmin(ldata),np.nanmax(ldata))
            else:
                vlims = (np.nanmin(ldata),np.nanmax(ldata))    
        else:
            vlims = (-np.nanmax(np.abs(ldata)),np.nanmax(np.abs(ldata)))
    
    return vlims


def get_transparent_cmap(alpha_mode='center', basemap='jet', vlims=None):
    
    if basemap == 'jet':
        bcmap = plt.cm.jet
    
    if basemap == 'bwr':
        bcmap = plt.cm.bwr
    
    cmap = bcmap(np.arange(bcmap.N))
    
    if alpha_mode == 'center':
        alpha = np.hstack((np.linspace(1, 0, int(bcmap.N/2)), np.linspace(0, 1, int(bcmap.N/2)))) 
    elif alpha_mode == 'top':
        alpha = np.linspace(1, 0, bcmap.N)
    elif alpha_mode == 'bottom':
        alpha = np.linspace(0, 1, bcmap.N)
    elif alpha_mode == 'real':
        sp = np.argwhere(np.diff(np.sign(np.linspace(vlims[0], vlims[1], bcmap.N))) == 2).reshape(-1)[0]
        alpha = np.hstack((np.linspace(1, 0, sp), np.linspace(0, 1, bcmap.N-sp))) 
        
    cmap[:,-1] = alpha

    return mpl.colors.ListedColormap(cmap)
        

def get_vlims_and_transparent_colormap(data, base_colormap='jet'):
    
    vlims = get_vlims(data, True)
    
    if np.prod(np.sign(vlims)) == -1:
        if np.abs(np.sort(vlims)[0]/np.sort(vlims)[1]) <= 0.25:
            cmap = get_transparent_cmap('real', base_colormap, vlims) 
        else:
            vlims = get_vlims(data, False)
            cmap = get_transparent_cmap('center', base_colormap)
    else:    
        if np.sum(np.sign(vlims)) <= 0:
            cmap = get_transparent_cmap('top', base_colormap)
        else:
            cmap = get_transparent_cmap('bottom', base_colormap)
            
    return vlims, cmap


def plot_power_cluster_summary(psds, freqs, sig_cl, clp, cl, T, info, legend=[]):
    
    # General font size
    mpl.rcParams["font.size"] = 10
    
    # Define mosaic
    mosaic = [['.' for j in range(9)] for i in range(2)]

    mosaic[0][0] = 'TOPO_1'
    mosaic[0][2] = 'TOPO_2'
    mosaic[0][4] = 'TOPO_3'
    mosaic[0][6] = 'TOPO_4'
    
    mosaic[0][8] = 'TOPO_CB'

    mosaic[1][0:9] = ['PLOT' for s in range(9)]    
    
    # Aspect ratio of mosaic
    w_ratios = [4,1,4,1,4,1,4,1,1]
    h_ratios = [1,2]
    
    # Create mosaic
    fig = plt.figure(constrained_layout=False, figsize=(10,10))
    axs = fig.subplot_mosaic(mosaic, height_ratios=h_ratios, width_ratios=w_ratios, gridspec_kw={'wspace':0.1, 'hspace':0.1}) 
    fig.set_size_inches([10,5])
    
    # Refine axes lines appearance
    [axs['TOPO_%d' % i].set_xticks([]) for i in range(1,5)]
    [axs['TOPO_%d' % i].set_yticks([]) for i in range(1,5)]
    [axs['TOPO_%d' % i].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,5)]
    
    axs['TOPO_CB'].spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    axs['TOPO_CB'].set_xticks([])
    axs['TOPO_CB'].set_yticks([])
    
    axs['PLOT'].spines[['right', 'top']].set_visible(False)
    axs['PLOT'].set_xlabel('Frequency [Hz]')
    axs['PLOT'].set_ylabel('EEG power')
    
    # All channels spectra
    axs['PLOT'].semilogy(freqs, np.mean(psds[0], axis=0), 'r--', linewidth=0.5)
    axs['PLOT'].semilogy(freqs, np.mean(psds[1], axis=0), 'b--', linewidth=0.5)
    
    if len(sig_cl) != 0:
        i = 1
        
        # vlims, cmap = get_vlims_and_transparent_colormap(T, 'jet')
        # vlims, cmap = get_vlims_and_transparent_colormap(np.mean(T[np.unique(np.hstack([np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1) for sc in sig_cl])),:], axis=0), 'jet')
        
        lT = np.mean((T*cl[sig_cl[0]])[np.argwhere(np.sum(cl[0], axis=1) != 0).reshape(-1),:], axis=0)
        for i in range(1,len(sig_cl)):
            lT = np.where(np.abs(lT) >= np.abs(np.mean((T*cl[sig_cl[i]])[np.argwhere(np.sum(cl[i], axis=1) != 0).reshape(-1),:], axis=0)), lT, np.mean((T*cl[sig_cl[0]])[np.argwhere(np.sum(cl[i], axis=1) != 0).reshape(-1),:], axis=0))
        
        vlims, cmap = get_vlims_and_transparent_colormap(lT, 'jet')
        
        for sc in sig_cl:
            
            c_idx = np.argwhere(np.sum(cl[sc], axis=0)).reshape(-1)
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            axs['PLOT'].semilogy(freqs[f_idx], np.mean(psds[0][c_idx,:][:,f_idx], axis=0), 'r', linewidth=2)
            axs['PLOT'].semilogy(freqs[f_idx], np.mean(psds[1][c_idx,:][:,f_idx], axis=0), 'b', linewidth=2)
            
            if i == 1:
                axs['PLOT'].legend(legend)
            
            axs['TOPO_%d' % i].set_title('T-value (%d)' % i)
            
            mask_ch = np.full(len(info['ch_names']), False)
            mask_ch[c_idx] = True
            
            im_S, _ = mne.viz.plot_topomap(    np.mean(T[f_idx,:], axis=0), info, 
                                                axes=axs['TOPO_%d' % i], cmap=cmap, vlim=vlims, contours=0, sensors=False, outlines='head',
                                                mask=mask_ch,
                                                mask_params=dict(marker='.', markeredgecolor='k', markersize=2)
                )
            
            i = i + 1
            
        
        cb_S = fig.colorbar(im_S, axs['TOPO_CB'])
        
        i = 1
        ymin, ymax = axs['PLOT'].get_ylim()
        for sc in sig_cl:
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            axs['PLOT'].fill_betweenx((ymin,ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.1)
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymin, "p = %.2g" % clp[sc], fontsize=10, ha='center', va='bottom')
            i = i + 1
    
   
    
    return fig, axs
        
        
    
    
def plot_pooled_power_cluster_summary(psds, sem_psds, freqs, sig_cl, clp, cl, T, legend=[]):
    
    # General font size
    mpl.rcParams["font.size"] = 10
    
    # Define mosaic
    mosaic = [['.' for j in range(9)] for i in range(1)]

    mosaic[0][0:9] = ['PLOT' for s in range(9)]    
    
    # Aspect ratio of mosaic
    w_ratios = [4,1,4,1,4,1,4,1,1]
    h_ratios = [1]
    
    # Create mosaic
    fig = plt.figure(constrained_layout=False, figsize=(10,10))
    axs = fig.subplot_mosaic(mosaic, height_ratios=h_ratios, width_ratios=w_ratios, gridspec_kw={'wspace':0.1, 'hspace':0.1}) 
    fig.set_size_inches([10,5])
    
    # Refine axes lines appearance    
    axs['PLOT'].spines[['right', 'top']].set_visible(False)
    axs['PLOT'].set_xlabel('Frequency [Hz]')
    axs['PLOT'].set_ylabel('EEG power')
    
    # All channels spectra
    axs['PLOT'].semilogy(freqs, psds[0], 'r--', linewidth=0.5)
    axs['PLOT'].semilogy(freqs, psds[1], 'b--', linewidth=0.5)
    
    # Clusters
    if len(sig_cl) != 0:
        i = 1
        ymin, ymax = axs['PLOT'].get_ylim()
        for sc in sig_cl:
            
            f_idx = np.argwhere(cl[sc]).reshape(-1)
            
            axs['PLOT'].semilogy(freqs[f_idx], psds[0][f_idx], 'r', linewidth=2)
            axs['PLOT'].semilogy(freqs[f_idx], psds[1][f_idx], 'b', linewidth=2)
            
            if i == 1:
                axs['PLOT'].legend(legend)
            
            axs['PLOT'].fill_betweenx((ymin,ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.1)
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "p = %.2g" % clp[sc], fontsize=10, ha='left', va='bottom', rotation=25)
            
            i = i + 1
        
    if len(sem_psds) != 0:
        axs['PLOT'].fill_between(freqs, psds[0] - sem_psds[0], psds[0] + sem_psds[0], alpha=0.1, color='r')
        axs['PLOT'].fill_between(freqs, psds[1] - sem_psds[1], psds[1] + sem_psds[1], alpha=0.1, color='b')

    return fig, axs
    
    

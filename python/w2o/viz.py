# Plotting utils

import numpy as np
import mne
import os
import time
import matplotlib.pyplot as plt

from w2o import filesystem
from w2o import utils
from w2o import preliminary

import matplotlib as mpl
import matplotlib.pyplot as plt

########### FUNCTIONS


def get_color_cycle():
    #return [vl for vl in mpl.colors.XKCD_COLORS.values()][0:20]
    return ['r', 'b', 'g', 'y', 'c', 'k', 'm', 'darkviolet', 'gray', 'tan', 'dodgerblue', 'navy']
    

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
        

def get_vlims_and_transparent_colormap(data, base_colormap='jet', real=False):
    
    vlims = get_vlims(data, True)
    
    if real:
        if (np.prod(np.sign(vlims)) == -1):
            cmap = get_transparent_cmap('real', base_colormap, vlims) 
        else:   
            if np.sum(np.sign(vlims)) <= 0:
                cmap = get_transparent_cmap('top', base_colormap)
            else:
                cmap = get_transparent_cmap('bottom', base_colormap)
        
        return vlims, cmap
    
    if np.prod(np.sign(vlims)) == -1:
        if np.abs(np.sort(vlims)[0]/np.sort(vlims)[1]) <= 0.2:
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
        
    
    
def plot_pooled_power_cluster_summary(spectra, sem_spectra, freqs, sig_cl, clp, cl, T, legend=[]):
    
    # Sort for increasing frequency start
    sig_cl = [sig_cl[j] for j in np.argsort([[i for i in range(len(freqs)) if cl[sc][i]][0] for sc in sig_cl])]
    
    # Number of conditions
    nC = len(spectra)
    
    # Colors
    clrs = get_color_cycle()[0:nC]
    
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
    for c in range(nC):
        axs['PLOT'].semilogy(freqs, spectra[c], '--', linewidth=0.5, color=clrs[c])
    
    # Clusters
    if len(sig_cl) != 0:
        i = 1
        ymin, ymax = axs['PLOT'].get_ylim()
        for sc in sig_cl:
            
            f_idx = np.argwhere(cl[sc]).reshape(-1)
            
            for c in range(nC):
                axs['PLOT'].semilogy(freqs[f_idx], spectra[c][f_idx], '', linewidth=2, color=clrs[c])
            
            if i == 1:
                axs['PLOT'].legend(legend)
            
            axs['PLOT'].fill_betweenx((ymin,ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.1)
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "p = %.2g" % clp[sc], fontsize=10, ha='left', va='bottom', rotation=25)
            
            i = i + 1
        
    if len(sem_spectra) != 0:
        for c in range(nC):
            axs['PLOT'].fill_between(freqs, spectra[c] - sem_spectra[c], spectra[c] + sem_spectra[c], alpha=0.1, color=clrs[c])

    return fig, axs
    
    
def plot_power_cluster_summary(spectra, sem_spectra, freqs, sig_cl, clp, cl, T, info, legend=[]):
    
    # Sort for increasing frequency start
    sig_cl = [sig_cl[j] for j in np.argsort([[i for i in range(len(freqs)) if np.sum(cl[sc], axis=1)[i]][0] for sc in sig_cl])]
    
    # General font size
    mpl.rcParams["font.size"] = 10
    
    # Number of conditions
    nC = len(spectra)
    
    # Colors
    clrs = get_color_cycle()[0:nC]
    
    # Mosaic size. Depends on n sig clusters
    m_size = [13, len(sig_cl)+1]
    
    # Define mosaic    
    mosaic = [['.' for j in range(m_size[0])] for i in range(m_size[1])]
    
    mosaic[m_size[1]-1][0:m_size[0]] = ['PLOT' for s in range(m_size[0])]
    
    for i in range(m_size[1]-1):
        for j in range(int(m_size[0]/3)):
            #print('(%d,%d) -> TOPO_%d_%d' % (i,3*j+1,m_size[1]-1-i,j+1))
            mosaic[i][3*j+1] = 'TOPO_%d_%d' % (m_size[1]-1-i,j+1)
            mosaic[i][3*j+2] = 'TOPO_CB_%d_%d' % (m_size[1]-1-i,j+1)
    
    # Aspect ratio of mosaic
    
    w_ratios = [1,4,1,1,4,1,1,4,1,1,4,1,1]
    h_ratios = [1 for i in range(m_size[1])]
    h_ratios[-1] = 3
    
    fig_size_inches = [9,4 + m_size[1]]
    
    # Create mosaic
    fig = plt.figure(constrained_layout=False, figsize=fig_size_inches)    
    axs = fig.subplot_mosaic(mosaic, height_ratios=h_ratios, width_ratios=w_ratios, gridspec_kw={'wspace':0.1, 'hspace':0.2}) 
    
    # Refine axes lines appearance
    [axs['TOPO_%d_%d' % (i,j)].set_xticks([]) for i in range(1,m_size[1]) for j in range(1,5)]
    [axs['TOPO_%d_%d' % (i,j)].set_yticks([]) for i in range(1,m_size[1]) for j in range(1,5)]
    [axs['TOPO_%d_%d' % (i,j)].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1]) for j in range(1,5)]
        
    [axs['TOPO_CB_%d_%d' % (i,j)].set_xticks([]) for i in range(1,m_size[1]) for j in range(1,5)]
    [axs['TOPO_CB_%d_%d' % (i,j)].set_yticks([]) for i in range(1,m_size[1]) for j in range(1,5)]
    [axs['TOPO_CB_%d_%d' % (i,j)].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1]) for j in range(1,5)]
        
    axs['PLOT'].spines[['right', 'top']].set_visible(False)
    axs['PLOT'].set_xlabel('Frequency [Hz]')
    axs['PLOT'].set_ylabel('EEG power')
    
    
    # Overall spectra with SEM
    for c in range(nC):
        axs['PLOT'].semilogy(freqs, np.mean(spectra[c], axis=0), '%s--' % clrs[c], linewidth=0.5)    
    
    axs['PLOT'].legend(legend)

    if len(sem_spectra) != 0:
        for c in range(nC):
            axs['PLOT'].fill_between(freqs, np.mean(spectra[c], axis=0) - sem_spectra[c], np.mean(spectra[c], axis=0) + sem_spectra[c], alpha=0.1, color=clrs[c])
    
    # Clusters info in main plot
    if len(sig_cl) != 0:
        
        i = 1        
        for sc in sig_cl:
            
            c_idx = np.argwhere(np.sum(cl[sc], axis=0) != 0).reshape(-1)
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            for c in range(nC):
                axs['PLOT'].semilogy(freqs[f_idx], np.mean(spectra[c][c_idx,:][:,f_idx], axis=0), clrs[c], linewidth=2)
            
            i = i + 1
        
        i = 1
        ymin, ymax = axs['PLOT'].get_ylim()
        for sc in sig_cl:
            
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            axs['PLOT'].fill_betweenx((ymin,ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.05)
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
            axs['PLOT'].text(np.mean(freqs[f_idx]), ymin, "p = %.2g" % clp[sc], fontsize=10, ha='center', va='bottom')
            
            i = i + 1
    
    
    # Clusters topoplots
    if len(sig_cl) != 0:
        
        i = 1        
        for sc in sig_cl:
            
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            if len(f_idx) <= 4:
                n_cols = len(f_idx)                
                fN = 1
            else:
                fN = int(np.ceil(len(f_idx) / 4))
                n_cols = int(len(f_idx)/fN)
                n_cols = int(np.ceil(len(f_idx)/fN))
            
            lTs = []
            for j in range(n_cols):
            
                lf_idx = f_idx[(fN*j):(fN*(j+1))]                
                lTs.append(np.mean(T[lf_idx,:], axis=0))
                
            vlims, cmap = get_vlims_and_transparent_colormap(np.asarray(lTs), 'jet', False)
            
            for j in range(n_cols):
                
                lf_idx = f_idx[(fN*j):(fN*(j+1))]                
                lc_idx = np.argwhere(np.sum(cl[sc][lf_idx,:], axis=0)).reshape(-1)
                                
                if nC <= 2:
                    axs['TOPO_%d_%d' % (i,j+1)].set_title('# %d - T-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx])), fontsize=8, pad=0)
                else:
                    axs['TOPO_%d_%d' % (i,j+1)].set_title('# %d - F-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx])), fontsize=8, pad=0)
                
                mask_ch = np.full(len(info['ch_names']), False)
                mask_ch[lc_idx] = True
                
                if np.mean(lTs[0][lc_idx]) < 0:
                    m_col = 'k'
                else:
                    m_col = 'k'
                    
                im_S, _ = mne.viz.plot_topomap(     lTs[j], info, 
                                                    axes=axs['TOPO_%d_%d' % (i,j+1)], cmap=cmap, vlim=vlims, contours=0, sensors=False, outlines='head',
                                                    mask=mask_ch,
                                                    mask_params=dict(marker='.', markeredgecolor=m_col, markersize=2)
                    )
            
                if j == n_cols - 1:
                    fig.colorbar(im_S, axs['TOPO_CB_%d_%d' % (i,4)])
            
            i = i +1
    
    
    
    return fig, axs


def plot_labels_power_cluster_summary(spectra, sem_spectra, freqs, sig_cl, clp, cl, T, labels, legend=[]):
    
    plt.ion()
    
    # Sort for increasing frequency start
    sig_cl = [sig_cl[j] for j in np.argsort([[i for i in range(len(freqs)) if np.sum(cl[sc], axis=1)[i]][0] for sc in sig_cl])]
    
    # General font size
    mpl.rcParams["font.size"] = 10
    
    # Number of conditions
    nC = len(spectra)
    
    # Splot labels on hemisphere
    lh_lidx = np.argwhere([labels[i].hemi == 'lh' for i in range(len(labels))]).reshape(-1)
    rh_lidx = np.argwhere([labels[i].hemi == 'rh' for i in range(len(labels))]).reshape(-1)
    
    # Get clusters hemispheres. Always disjointed (no interhemispheric connectivity)
    if len(sig_cl) != 0:
        hemis = []
        for sc in sig_cl:
            assert len(np.unique([labels[l].hemi for l in np.argwhere(np.sum(cl[sc], axis=0) != 0).reshape(-1)])) == 1
            hemis.append(np.unique([labels[l].hemi for l in np.argwhere(np.sum(cl[sc], axis=0) != 0).reshape(-1)])[0])
    
    # Get fsaverage brain dimensions
    src = mne.read_source_spaces(os.path.join(filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    n_verts = {'lh': src[0]['nn'].shape[0], 'rh': src[1]['nn'].shape[0]}
    del src
    
    # Colors
    clrs = get_color_cycle()[0:nC]
    
    # Mosaic size. Depends on n sig clusters
    m_size = [13, len(sig_cl)+2]
    
    # Define mosaic    
    mosaic = [['.' for j in range(m_size[0])] for i in range(m_size[1])]
    
    mosaic[m_size[1]-2][0:m_size[0]] = ['PLOT_L' for s in range(m_size[0])]
    mosaic[m_size[1]-1][0:m_size[0]] = ['PLOT_R' for s in range(m_size[0])]
    
    for i in range(m_size[1]-2):
        for j in range(int(m_size[0]/3)):
            #print('(%d,%d) -> SRC_%d_%d' % (i,3*j+1,m_size[1]-2-i,j+1))
            mosaic[i][3*j+1] = 'SRC_%d_%d' % (m_size[1]-2-i,j+1)
            mosaic[i][3*j+2] = 'SRC_CB_%d_%d' % (m_size[1]-2-i,j+1)
    
    # Aspect ratio of mosaic    
    w_ratios = [1,4,1,1,4,1,1,4,1,1,4,1,1]
    h_ratios = [1 for i in range(m_size[1])]
    h_ratios[-2] = 1.5
    h_ratios[-1] = 1.5
    
    fig_size_inches = [12,6 + m_size[1]]
    
    # Create mosaic
    fig = plt.figure(constrained_layout=False, figsize=fig_size_inches)    
    axs = fig.subplot_mosaic(mosaic, height_ratios=h_ratios, width_ratios=w_ratios, gridspec_kw={'wspace':0.1, 'hspace':0.2}) 
    
    # Refine axes lines appearance
    [axs['SRC_%d_%d' % (i,j)].set_xticks([]) for i in range(1,m_size[1]-1) for j in range(1,5)]
    [axs['SRC_%d_%d' % (i,j)].set_yticks([]) for i in range(1,m_size[1]-1) for j in range(1,5)]
    [axs['SRC_%d_%d' % (i,j)].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1]-1) for j in range(1,5)]
        
    [axs['SRC_CB_%d_%d' % (i,j)].set_xticks([]) for i in range(1,m_size[1]-1) for j in range(1,5)]
    [axs['SRC_CB_%d_%d' % (i,j)].set_yticks([]) for i in range(1,m_size[1]-1) for j in range(1,5)]
    [axs['SRC_CB_%d_%d' % (i,j)].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1]-1) for j in range(1,5)]
        
    axs['PLOT_L'].spines[['right', 'top']].set_visible(False)
    axs['PLOT_L'].set_xlabel('')
    axs['PLOT_L'].set_ylabel('Source power (Left)')
    
    axs['PLOT_R'].spines[['right', 'top']].set_visible(False)
    axs['PLOT_R'].set_xlabel('Frequency [Hz]')
    axs['PLOT_R'].set_ylabel('Source power (Right)')
    
    # Overall spectra with SEM
    for c in range(nC):
        axs['PLOT_L'].semilogy(freqs, np.mean(spectra[c][lh_lidx,:], axis=0), '%s--' % clrs[c], linewidth=0.5)    
        axs['PLOT_R'].semilogy(freqs, np.mean(spectra[c][rh_lidx,:], axis=0), '%s--' % clrs[c], linewidth=0.5)    
    
    axs['PLOT_R'].legend(legend, ncol=2)

    if len(sem_spectra) != 0:
        for c in range(nC):
            axs['PLOT_L'].fill_between(freqs, np.mean(spectra[c][lh_lidx,:], axis=0) - np.mean(sem_spectra[c][lh_lidx,:], axis=0), np.mean(spectra[c][lh_lidx,:], axis=0) + np.mean(sem_spectra[c][lh_lidx,:], axis=0), alpha=0.1, color=clrs[c])
            axs['PLOT_R'].fill_between(freqs, np.mean(spectra[c][rh_lidx,:], axis=0) - np.mean(sem_spectra[c][rh_lidx,:], axis=0), np.mean(spectra[c][rh_lidx,:], axis=0) + np.mean(sem_spectra[c][rh_lidx,:], axis=0), alpha=0.1, color=clrs[c])
    
    
    # Clusters info in main plot
    if len(sig_cl) != 0:
        
        i = 1        
        for sc in sig_cl:
            
            l_idx = np.argwhere(np.sum(cl[sc], axis=0) != 0).reshape(-1)
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            for c in range(nC):
                if hemis[i-1] == 'lh':
                    axs['PLOT_L'].semilogy(freqs[f_idx], np.mean(spectra[c][l_idx,:][:,f_idx], axis=0), clrs[c], linewidth=2)
                if hemis[i-1] == 'rh':
                    axs['PLOT_R'].semilogy(freqs[f_idx], np.mean(spectra[c][l_idx,:][:,f_idx], axis=0), clrs[c], linewidth=2)
            
            i = i + 1
        
        i = 1
        l_ymin, l_ymax = axs['PLOT_L'].get_ylim()
        r_ymin, r_ymax = axs['PLOT_R'].get_ylim()
        for sc in sig_cl:
            
            if hemis[i-1] == 'lh':
                f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
                axs['PLOT_L'].fill_betweenx((l_ymin,l_ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.05)
                axs['PLOT_L'].text(np.mean(freqs[f_idx]), l_ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
                axs['PLOT_L'].text(np.mean(freqs[f_idx]), l_ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
                axs['PLOT_L'].text(np.mean(freqs[f_idx]), l_ymin, "p = %.2g" % clp[sc], fontsize=10, ha='center', va='bottom')
            
            if hemis[i-1] == 'rh':
                f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
                axs['PLOT_R'].fill_betweenx((r_ymin,r_ymax), freqs[f_idx[0]], freqs[f_idx[-1]], color="gray", alpha=0.05)
                axs['PLOT_R'].text(np.mean(freqs[f_idx]), r_ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
                axs['PLOT_R'].text(np.mean(freqs[f_idx]), r_ymax, "%d" % i, fontsize=10, ha='center', va='bottom')
                axs['PLOT_R'].text(np.mean(freqs[f_idx]), r_ymin, "p = %.2g" % clp[sc], fontsize=10, ha='center', va='bottom')
            
            i = i + 1
    
    # Clusters brain flat maps
    if len(sig_cl) != 0:
        
        i = 1        
        for sc in sig_cl:
            
            f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
            
            if len(f_idx) <= 4:
                n_cols = len(f_idx)                
                fN = 1
            else:
                fN = int(np.ceil(len(f_idx) / 4))
                n_cols = int(len(f_idx)/fN)
                n_cols = int(np.ceil(len(f_idx)/fN))
            
            lTs = []
            lTranges = []
            for j in range(n_cols):
            
                lf_idx = f_idx[(fN*j):(fN*(j+1))]    
                ll_idx = np.argwhere(np.sum(cl[sc][lf_idx,:], axis=0)).reshape(-1)
                lTs.append(np.mean(T[lf_idx,:], axis=0))
                lTranges.append([np.min(np.mean(T[lf_idx,:][:,ll_idx], axis=0)), np.max(np.mean(T[lf_idx,:][:,ll_idx], axis=0))])
                
            vlims, cmap = get_vlims_and_transparent_colormap(np.asarray(lTranges), 'jet', True)
            
            for j in range(n_cols):
                
                lf_idx = f_idx[(fN*j):(fN*(j+1))]                
                ll_idx = np.argwhere(np.sum(cl[sc][lf_idx,:], axis=0)).reshape(-1)
                                
                if nC <= 2:
                    axs['SRC_%d_%d' % (i,j+1)].set_title('# %d - T-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx])), fontsize=8, pad=0)
                else:
                    axs['SRC_%d_%d' % (i,j+1)].set_title('# %d - F-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx])), fontsize=8, pad=0)
                                
                
                brain = mne.viz.Brain('fsaverage', hemi=hemis[i-1], surf='flat', size=200, background='white')            
                bdata = np.zeros(n_verts[hemis[i-1]])
                for ll in ll_idx:
                    bdata[labels[ll].vertices] = lTs[j][ll]
                brain.add_data(bdata, fmin=vlims[0], fmax=vlims[1], colormap=cmap, colorbar=False)
                screenshot = brain.screenshot()
                screenshot = screenshot[(screenshot != 255).any(-1).any(1)][:,(screenshot != 255).any(-1).any(0)]
                time.sleep(0.01)
                axs['SRC_%d_%d' % (i,j+1)].imshow(screenshot)    
                time.sleep(0.01)
                brain.close()
                
                if j == n_cols - 1:
                    mne.viz.plot_brain_colorbar(axs['SRC_CB_%d_%d' % (i,4)], {'kind' : 'value', 'lims' : [vlims[0], np.mean(vlims), vlims[1]]}, colormap=cmap, transparent=True, orientation='vertical', label=None, bgcolor='w')
            
            i = i +1
   
    # Onclick callback
    def onclick(event):
        
        axlabel = event.inaxes.get_label()
        i = int(axlabel[4:5])        
        
        sc = sig_cl[i-1]
        
        f_idx = np.argwhere(np.sum(cl[sc], axis=1) != 0).reshape(-1)
        
        if len(f_idx) <= 4:
            n_cols = len(f_idx)                
            fN = 1
        else:
            fN = int(np.ceil(len(f_idx) / 4))
            n_cols = int(len(f_idx)/fN)
            n_cols = int(np.ceil(len(f_idx)/fN))
        
        lTs = []
        lTranges = []
        for j in range(n_cols):
        
            lf_idx = f_idx[(fN*j):(fN*(j+1))]    
            ll_idx = np.argwhere(np.sum(cl[sc][lf_idx,:], axis=0)).reshape(-1)
            lTs.append(np.mean(T[lf_idx,:], axis=0))
            lTranges.append([np.min(np.mean(T[lf_idx,:][:,ll_idx], axis=0)), np.max(np.mean(T[lf_idx,:][:,ll_idx], axis=0))])
            
        vlims, cmap = get_vlims_and_transparent_colormap(np.asarray(lTranges), 'jet', True)
        
        j = int(axlabel[6:7]) - 1
        
        lf_idx = f_idx[(fN*j):(fN*(j+1))]                
        ll_idx = np.argwhere(np.sum(cl[sc][lf_idx,:], axis=0)).reshape(-1)
        
        if nC <= 2:
            btitle = '# %d - T-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx]))
        else:
            btitle = '# %d - F-Value (%.1f Hz)' % (i, np.mean(freqs[lf_idx]))
        
        brain = mne.viz.Brain('fsaverage', hemi=hemis[i-1], surf='inflated', size=600, background='white', title=btitle)
        bdata = np.zeros(n_verts[hemis[i-1]])
        for ll in ll_idx:
            bdata[labels[ll].vertices] = lTs[j][ll]
        brain.add_data(bdata, fmin=vlims[0], fmax=vlims[1], colormap=cmap, colorbar=True)
        
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    

def plot_fbands_power_cluster_summary(fb_spectra, sig_cl, clp, cl, T, info, conditions=[]):
    
    # General font size
    mpl.rcParams["font.size"] = 10
    
    # Number of conditions
    nC = len(fb_spectra)
    
    # Sample size
    N = len(fb_spectra[0])
    
    # Colors
    clrs = get_color_cycle()[0:nC]
    
    # Mosaic size. Depends on n sig clusters
    m_size = [5, len(sig_cl)+1]
    
    # Define mosaic    
    mosaic = [['.' for j in range(m_size[0])] for i in range(m_size[1])]
    
    mosaic[m_size[1]-1][3] = 'VIOLIN_ALL'
    mosaic[m_size[1]-1][4] = 'PTS_ALL'
    
    for i in range(m_size[1]-1):
        mosaic[i][0] ='TOPO_CB_%d' % (m_size[1]-1-i)
        mosaic[i][1] ='TOPO_%d' % (m_size[1]-1-i)
        mosaic[i][3] ='VIOLIN_%d' % (m_size[1]-1-i)
        mosaic[i][4] ='PTS_%d' % (m_size[1]-1-i)
            
    
    # Aspect ratio of mosaic    
    w_ratios = [0.5,3,1.5,6,6]
    h_ratios = [1 for i in range(m_size[1])]
    
    fig_size_inches = [12,2 + 2*m_size[1]]
    
    # Create mosaic
    fig = plt.figure(constrained_layout=False, figsize=fig_size_inches)    
    axs = fig.subplot_mosaic(mosaic, height_ratios=h_ratios, width_ratios=w_ratios, gridspec_kw={'wspace':0.1, 'hspace':0.1}) 
    
    # Refine axes lines appearance
    [axs['TOPO_CB_%d' % i].set_xticks([]) for i in range(1,m_size[1])]
    [axs['TOPO_CB_%d' % i].set_yticks([]) for i in range(1,m_size[1])]
    [axs['TOPO_CB_%d' % i].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1])]
    
    [axs['TOPO_%d' % i].set_xticks([]) for i in range(1,m_size[1])]
    [axs['TOPO_%d' % i].set_yticks([]) for i in range(1,m_size[1])]
    [axs['TOPO_%d' % i].spines[['right', 'top', 'left', 'bottom']].set_visible(False) for i in range(1,m_size[1])]
        
    [axs['VIOLIN_%d' % i].spines[['right', 'top']].set_visible(False) for i in range(1,m_size[1])]
    [axs['VIOLIN_%d' % i].set_xticklabels([]) for i in range(1,m_size[1])]
    
    [axs['PTS_%d' % i].spines[['left', 'top']].set_visible(False) for i in range(1,m_size[1])]
    [axs['PTS_%d' % i].yaxis.tick_right() for i in range(1,m_size[1])]
    [axs['PTS_%d' % i].set_xticklabels([]) for i in range(1,m_size[1])]
    
    axs['VIOLIN_ALL'].spines[['right', 'top']].set_visible(False)
    
    axs['PTS_ALL'].spines[['left', 'top']].set_visible(False)
    axs['PTS_ALL'].yaxis.tick_right()
    
    
    # Violin plots of all channels
    X= np.linspace(1,nC,nC)    
    
    ldata = [[np.mean(fb_spectra[c][s]) for s in range(N)] for c in range(nC)]
    
    [axs['VIOLIN_ALL'].plot(X[c], np.mean([ldata[c][s] for s in range(N)]), 'o', color=clrs[c]) for c in range(nC)]
    [axs['VIOLIN_ALL'].plot(X[c], np.median([ldata[c][s] for s in range(N)]), 'x', color=clrs[c]) for c in range(nC)]
    
    vln_df = axs['VIOLIN_ALL'].violinplot([np.asarray([ldata[c][s] for s in range(N)]) for c in range(nC)], X, showextrema=False, showmeans=False, showmedians=False)
    [vb.set_color(clrs[i]) for i,vb in enumerate(vln_df['bodies'])]
    [vb.set_alpha(0.15) for vb in vln_df['bodies']]
    axs['VIOLIN_ALL'].set_xticks(X)
    axs['VIOLIN_ALL'].set_xticklabels(conditions, rotation=30, ha='right')
    axs['VIOLIN_ALL'].set_yscale('log')
    axs['VIOLIN_ALL'].set_ylabel('EEG Power')
    
    [[axs['PTS_ALL'].plot(X[c:(c+2)], [ldata[c][s], ldata[c+1][s]], 'k--', alpha=0.3, linewidth=0.5) for s in range(N)] for c in range(nC-1)]
    [[axs['PTS_ALL'].plot(X[c], ldata[c][s], '+', color=clrs[c], alpha=0.3) for s in range(N)] for c in range(nC)]
    [axs['PTS_ALL'].plot(X[c:(c+2)], [np.mean([ldata[c][s] for s in range(N)]), np.mean([ldata[c+1][s] for s in range(N)])], 'k-') for c in range(nC-1)]
    [axs['PTS_ALL'].plot(X[c], np.mean([ldata[c][s] for s in range(N)]), 'o', color=clrs[c]) for c in range(nC)]
    axs['PTS_ALL'].set_yscale('log')
    axs['PTS_ALL'].set_xlim([X[0]-0.25, X[-1]+0.25])
    axs['PTS_ALL'].set_xticks(X)
    axs['PTS_ALL'].set_xticklabels(conditions, rotation=30, ha='right')
    
    # Clusters
    i = 1
    vlims, cmap = get_vlims_and_transparent_colormap(np.asarray(T), 'jet', False)
    for sc in sig_cl:
        
        c_idx = np.argwhere(cl[sc] != 0).reshape(-1)
        
        ldata = [[np.mean(fb_spectra[c][s][c_idx]) for s in range(N)] for c in range(nC)]
        
        [axs['VIOLIN_%d' % i].plot(X[c], np.mean([ldata[c][s] for s in range(N)]), 'o', color=clrs[c]) for c in range(nC)]
        [axs['VIOLIN_%d' % i].plot(X[c], np.median([ldata[c][s] for s in range(N)]), 'x', color=clrs[c]) for c in range(nC)]
        
        vln_df = axs['VIOLIN_%d' % i].violinplot([np.asarray([ldata[c][s] for s in range(N)]) for c in range(nC)], X, showextrema=False, showmeans=False, showmedians=False)
        [vb.set_color(clrs[i]) for i,vb in enumerate(vln_df['bodies'])]
        [vb.set_alpha(0.15) for vb in vln_df['bodies']]
        axs['VIOLIN_%d' % i].set_xticks(X)        
        axs['VIOLIN_%d' % i].set_yscale('log')
        axs['VIOLIN_%d' % i].set_ylabel('EEG Power')
        
        [[axs['PTS_%d' % i].plot(X[c:(c+2)], [ldata[c][s], ldata[c+1][s]], 'k--', alpha=0.3, linewidth=0.5) for s in range(N)] for c in range(nC-1)]
        [[axs['PTS_%d' % i].plot(X[c], ldata[c][s], '+', color=clrs[c], alpha=0.3) for s in range(N)] for c in range(nC)]
        [axs['PTS_%d' % i].plot(X[c:(c+2)], [np.mean([ldata[c][s] for s in range(N)]), np.mean([ldata[c+1][s] for s in range(N)])], 'k-') for c in range(nC-1)]
        [axs['PTS_%d' % i].plot(X[c], np.mean([ldata[c][s] for s in range(N)]), 'o', color=clrs[c]) for c in range(nC)]
        axs['PTS_%d' % i].set_yscale('log')
        axs['PTS_%d' % i].set_xlim([X[0]-0.25, X[-1]+0.25])
        axs['PTS_%d' % i].set_xticks(X)
        
        mask_ch = np.full(len(info['ch_names']), False)
        mask_ch[c_idx] = True
        
        if np.mean(T[c_idx]) < 0:
            m_col = 'k'
        else:
            m_col = 'k'
        
        im_S, _ = mne.viz.plot_topomap(     T, info, 
                                            axes=axs['TOPO_%d' % i], cmap=cmap, vlim=vlims, contours=0, sensors=False, outlines='head',
                                            mask=mask_ch,
                                            mask_params=dict(marker='.', markeredgecolor=m_col, markersize=2)
            )
    
        fig.colorbar(im_S, axs['TOPO_CB_%d' % i])       
        
        axs['TOPO_CB_%d' % i].yaxis.set_ticks_position('left')
        if nC <= 2:
            axs['TOPO_%d' % i].set_title('T-Value (p = %.2g)' % clp[sc], fontsize=10, pad=0)
        else:
            axs['TOPO_%d' % i].set_title('F-Value (p = %.2g)' % clp[sc], fontsize=10, pad=0)
        
        i = i + 1
    
    return fig, axs
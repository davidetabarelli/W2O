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


# Utilities

import socket
import mne;
import numpy as np;
import matplotlib 
import matplotlib.pyplot as plt



#### FUNCTIONS

def get_fbands_dict():
    
    fbands_dict = {}
    
    fbands_dict['Theta'] = [4, 7]
    fbands_dict['MuAlpha'] = [8, 13]
    fbands_dict['Beta'] = [15, 30]
    fbands_dict['LowGamma'] = [35, 48] 
    fbands_dict['HighGamma'] = [65, 85] 
    fbands_dict['HighFreq'] = [90, 180]     
    
    return fbands_dict


def get_njobs():

    if socket.gethostname()[:18] == 'MacBook-Pro-Davide':
        return 16
    
    if socket.gethostname()[:12] == 'meg-server-4':
        return 64
    
    if socket.gethostname()[:13] == 'irbio-server1':        
        return 130
    
    if socket.gethostname()[:13] == 'irbio-server2':        
        return 130
    
    if socket.gethostname()[:13] == 'irbio-server3':        
        return 130

def get_colors_list():
    
    return list(matplotlib.colors.BASE_COLORS.keys())
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


    
def ref_to_avg_with_fcz(raw, proj=False):
    
    rraw = raw.copy()
       
    # Re-reference to average and add FCz
    rraw.add_reference_channels('FCz')
    rraw.set_eeg_reference('average', ch_type='eeg', projection=proj)
    rraw.apply_proj()
    rraw.set_montage('standard_1005')
    
    return rraw
    

# Requires average reference data with FCz reconstructed
def double_banana_ref(raw, reverse=False):
    
    rraw = raw.copy()
    
    if not np.any(np.isin(np.asarray(rraw.info['ch_names']), 'FCz')):
        rraw = ref_to_avg_with_fcz(rraw)
        
    if reverse:
        anode = []
        cathode = []
    else:
        #anode =     ['Fp1', 'AF7', 'F7', 'FT7', 'T7', 'TP7', 'P7', 'PO7', 'Fp1', 'F5',  'FC5', 'C5',  'CP5', 'P5',  'PO3', 'Fp1', 'AF3', 'F3',  'FC3', 'C3',  'CP3', 'P3', 'Fp1', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Fp2', 'F2', 'FC2', 'C2', 'CP2', 'P2', 'Fp2', 'AF4', 'F4',  'FC4', 'C4', 'CP4', 'P4', 'PO4', 'Fp2', 'F6', 'FC6', 'C6', 'CP6', 'P6',  'Fp2', 'AF8', 'F8',  'FT8', 'T8', 'TP8', 'P8', 'PO8']
        #cathode =   ['AF7', 'F7',  'FT7', 'T7', 'TP7', 'P7', 'PO7', 'O1', 'F5',  'FC5', 'C5',  'CP5', 'P5',  'PO3', 'O1',  'AF3', 'F3',  'FC3', 'C3',  'CP3', 'P3',  'PO3', 'F1', 'FC1', 'C1', 'CP1', 'P1', 'O1', 'Fz',  'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'F2', 'FC2', 'C2',  'CP2', 'P2', 'O2', 'AF4', 'F4',  'FC4', 'C4',  'CP4', 'P4', 'PO4', 'O2', 'F6',  'FC6', 'C6', 'CP6', 'P6', 'PO4', 'AF8', 'F8',  'FT8', 'T8',  'TP8', 'P8', 'PO8', 'O2']
        anode =     ['AF7', 'F7', 'FT7', 'T7', 'TP7', 'P7',  'F5',  'FC5', 'C5',  'CP5',  'PO3', 'Fp1', 'AF3', 'F3',  'FC3', 'C3',  'CP3', 'P3', 'F1', 'FC1', 'C1', 'CP1', 'AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'F2', 'FC2', 'C2', 'CP2', 'Fp2', 'AF4', 'F4',  'FC4', 'C4', 'CP4', 'P4', 'PO4', 'F6', 'FC6', 'C6', 'CP6', 'AF8', 'F8',  'FT8', 'T8', 'TP8', 'P8']
        cathode =   ['F7',  'FT7', 'T7', 'TP7', 'P7', 'PO7', 'FC5', 'C5',  'CP5', 'P5', 'O1',  'AF3', 'F3',  'FC3', 'C3',  'CP3', 'P3',  'PO3', 'FC1', 'C1', 'CP1', 'P1', 'Fz',  'FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz', 'FC2', 'C2',  'CP2', 'P2', 'AF4', 'F4',  'FC4', 'C4',  'CP4', 'P4', 'PO4', 'O2', 'FC6', 'C6', 'CP6', 'P6', 'F8',  'FT8', 'T8',  'TP8', 'P8', 'PO8']
    
    dbraw = mne.set_bipolar_reference(rraw, anode=anode, cathode=cathode, copy=True)
    
    # TODO: set positions according to average between channels position
    
    return dbraw

# Utilities

import mne;
import numpy as np;
import matplotlib 
import matplotlib.pyplot as plt

from w2o import preliminary
from w2o import dataset
from w2o import utils



#### FUNCTIONS


# Frequency bands definition
def get_fbands_dict(mode='standard'):
    
    fbands_dict = {}
    
    # fbands_dict['Delta'] = [2, 4]    
    # fbands_dict['Theta'] = [4, 7]
    # fbands_dict['Alpha'] = [8, 13]
    # fbands_dict['Beta'] = [15, 30]
    # fbands_dict['Gamma'] = [31, 48] 
    
    if mode == 'standard':
        fbands_dict['Delta'] = [2, 4]    
        fbands_dict['Theta'] = [4, 7]
        fbands_dict['Alpha'] = [8, 12]
        fbands_dict['Beta'] = [13, 30]
        fbands_dict['Gamma'] = [31, 40] 
    if mode == 'data_driven':
        fbands_dict['Delta'] = [2, 4]    
        fbands_dict['Theta'] = [4, 7]
        fbands_dict['Alpha'] = [8, 13]
        fbands_dict['Beta'] = [16, 27]
        fbands_dict['Gamma'] = [31, 44]         
        
    
    return fbands_dict



# Compute spectra of periods, depending on periods
# IN: p_raws data, periods selection or []
# OPTION: normalize/not on norm_period = 'FixRest'
# OUT: p_psds, avg_p_psds, fbands
def get_periods_psds(subject, periods=[], norm_period=[]):
    
    # Get periods of interest
    if periods == []:
        iperiods = [k for k in dataset.get_periods_definition().keys()]
    else:
        iperiods = periods
    
    
    # Frequency bands
    f_bands = get_fbands_dict()
    
    # Get craw
    craw, events, evt_dict = preliminary.get_clean_data(subject, True)
    
    # Pick EEG channels
    craw.pick('eeg')
    
    # Get periods
    p_raws = preliminary.extract_periods(craw, events, evt_dict, 0, iperiods)

    # Create epochs
    p_epochs = {}
    for ip in p_raws.keys():
        
        if ip == 'Muscles':
            
            lepochs = preliminary.extract_muscles_epochs(p_raws[ip], events, evt_dict, 2.0, 0.5)
                    
        elif np.isin(ip, ['VibTest', 'Vib1', 'Vib2']):
            
            lepochs = preliminary.extract_vib_epochs(p_raws[ip], events, evt_dict)
            
        else:
            lepochs = mne.make_fixed_length_epochs(p_raws[ip], duration=2, proj=True, reject_by_annotation=True, overlap=0.5)
            lepochs.drop_bad()
        
        p_epochs[ip] = lepochs
    
    
    # Join Vibs periods adding a new one
    if np.all(np.isin(['VibTest', 'Vib1', 'Vib2'], iperiods)):
        p_epochs['VibOn'] = mne.concatenate_epochs([p_epochs[ip]['VibOn'] for ip in ['VibTest', 'Vib1', 'Vib2'] if p_epochs[ip] != None])
        p_epochs['VibOff'] = mne.concatenate_epochs([p_epochs[ip]['VibOff'] for ip in ['VibTest', 'Vib1', 'Vib2'] if p_epochs[ip] != None])
    
        # Prune old Vib epochs
        [p_epochs.pop(vp) for vp in ['VibTest', 'Vib1', 'Vib2']]
    
    
    # Compute PDS
    p_psds = {}
    freqs = {}
    for ip in p_epochs.keys():
        
        if np.isin(ip, ['VibOn', 'VibOff']):
            p_psds[ip] = p_epochs[ip].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=utils.get_njobs(), adaptive=False, low_bias=False)            
        else:
            p_psds[ip] = p_epochs[ip].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=utils.get_njobs(), adaptive=False, low_bias=False)
        
        freqs[ip] = p_psds[ip].freqs
        
    
    
    # If selected normalize PSD on selected normalization period 
    if norm_period != []:
        for ip in p_psds.keys():
            ff_idx = np.argwhere(np.isin(p_psds[norm_period].freqs, p_psds[ip].freqs)).reshape(-1)
            bsln = p_psds[norm_period].copy().average().get_data()[:,ff_idx]
            ndata = (p_psds[ip].get_data() / bsln[np.newaxis,:,:])
            p_psds[ip] = mne.time_frequency.EpochsSpectrumArray(ndata, p_psds[ip].info, p_psds[ip].freqs)
    
    # Average PSD
    avg_p_psds = {ip : p_psds[ip].copy().average() for ip in p_psds.keys()}
    
    
    # Create fbands. Created from averaged (and eventually normalized) spectra
    fb_p_psds = {}
    for ip in p_epochs.keys():
        fb_p_psds[ip] = {fb: None for fb in f_bands.keys()}
        for fb in f_bands.keys():
            
            fmin = f_bands[fb][0]
            fmax = f_bands[fb][1]
            
            f_idx = np.argwhere((freqs[ip] >= fmin) & (freqs[ip] <= fmax)).reshape(-1)
            
            fb_p_psds[ip][fb] = np.mean(avg_p_psds[ip].get_data()[:, f_idx], axis=1)
        
    
    return p_psds, avg_p_psds, fb_p_psds, freqs
# Get all datasets info and save a dataframe

import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()

periods = w2o.dataset.get_periods_definition().keys()

info = pd.DataFrame({ ** { 
                            'Subject': pd.Series(dtype='str'), 
                            'Bad_Channels': pd.Series(dtype='int'), 
                            'Total_Comps': pd.Series(dtype='int'), 
                            'Rejected_Comps': pd.Series(dtype='int')
                        },
                      ** {'%s_Time' % period :pd.Series(dtype='float') for period in periods}                      
                    })

for subject in subjects:
    
    nrow = {}
    
    nrow['Subject'] = subject
    
    bads = np.loadtxt(fname=os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-badchannels.txt' % subject), dtype='bytes').astype(str).tolist();
    if bads == []:
        nrow['Bad_Channels'] = 0
    elif type(bads) is list:
        nrow['Bad_Channels'] = 1
    else:
        nrow['Bad_Channels'] = len(bads)
    
    ica = mne.preprocessing.read_ica(os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject))
    
    nrow['Total_Comps'] = ica.n_components_
    
    ica_excluded = np.loadtxt(fname=os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-ica-excluded.txt' % subject), dtype='bytes').astype(int).tolist();
    ica_enhc_excluded = np.loadtxt(fname=os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-ica-enhc-excluded.txt' % subject), dtype='bytes').astype(int).tolist();
    
    nrow['Rejected_Comps'] = len(ica_excluded + ica_enhc_excluded)
    
    craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
    p_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 0.0)
    
    for period in periods:
        nrow['%s_Time' % period] = (p_raws[period].last_samp - p_raws[period].first_samp) / p_raws[period].info['sfreq']
        
    info = pd.concat([info, pd.DataFrame([nrow])], ignore_index=True)


# Save



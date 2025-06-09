import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()
tN = 19


# Get all subjects data divided into all periods
p_raws = [];
db_p_raws = [];
for subject in subjects[0:tN]:
    craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
    dbraw = w2o.utils.double_banana_ref(craw)    
    p_raws.append(w2o.preliminary.extract_periods(craw, events, evt_dict, 0))
    db_p_raws.append(w2o.preliminary.extract_periods(dbraw, events, evt_dict, 0))



# Compute all average power in frequency bands in periods of interests
#iperiods = ['FixRest', 'EcRest', 'Porn', 'Masturbation', 'Pleateau', 'Orgasm']
#iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm']
iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Breathe1', 'Breathe2', 'FixRest', 'Porn']
normPeriod = 'FixRest'
fbands = w2o.dataset.get_fbands_dict()
fb_df = pd.DataFrame({  
                        'Subject': pd.Series(dtype='str'), 
                        'Band': pd.Series(dtype='str'), 
                        'Period': pd.Series(dtype='str'), 
                        'Power': pd.Series(dtype='float'), 
                        'nPower': pd.Series(dtype='float'),     # Normalized to FixRest
                    })

for s,subject in enumerate(subjects[0:tN]):
    
    for period in iperiods:
        
        psd = mne.make_fixed_length_epochs(p_raws[s][period], duration=1.5, proj=True, reject_by_annotation=True, overlap=0.375).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
   
        freqs = psd.freqs
        power = psd.get_data()
        del psd
    
        for band, fb in fbands.items():
            
            fb_idx = np.argwhere((freqs >= fb[0]) & (freqs <= fb[1])).reshape(-1)
            
            nrow ={}
            
            nrow['Subject'] = subject
            nrow['Band'] = band
            nrow['Period'] = period
            #nrow['Power'] = np.mean(np.mean(np.sqrt(power), axis=0)[fb_idx]) # SQRT ????
            nrow['Power'] = np.mean(np.mean(power, axis=0)[fb_idx])
            nrow['nPower'] = 0.0
            
            fb_df = pd.concat([fb_df, pd.DataFrame([nrow])], ignore_index=True)

for subject in subjects[0:tN]:
    for band, fb in fbands.items():
        normP = fb_df.query('Subject == @subject').query('Band==@band').query('Period==@normPeriod')['Power'].values[0]
        for period in iperiods:
            fb_df.loc[fb_df.query('Subject == @subject').query('Band==@band').query('Period==@period').index, 'nPower'] = (fb_df.query('Subject == @subject').query('Band==@band').query('Period==@period')['Power'].values[0] / normP) - 1.0
        

# Vibs
# vib_on_epochs = mne.Epochs(craw, mne.merge_events(events, [evt_dict['Vib_test_on'], evt_dict['Vib_1_on'], evt_dict['Vib_2_on']], 200), 200, tmin=0.0, tmax=1.0, baseline=(None,1.0), picks='eeg', reject_by_annotation=True, proj=True)
# vib_off_epochs = mne.Epochs(craw, mne.merge_events(events, [evt_dict['Vib_test_off'], evt_dict['Vib_1_off'], evt_dict['Vib_2_off']], 201), 201, 0, 1.0, baseline=(None,1.0), picks='eeg', reject_by_annotation=True, proj=True)

# vib_on_epochs.drop_bad()
# vib_off_epochs.drop_bad()

# vib_on_psd = vib_on_epochs.compute_psd(method='welch', fmin=1.0, fmax=95, n_fft=500, proj=True).average()
# vib_off_psd = vib_off_epochs.compute_psd(method='welch', fmin=1.0, fmax=95, n_fft=500, proj=True).average()

# freqs = vib_on_psd.freqs

# plt.loglog(freqs, np.mean(vib_on_psd.get_data(), axis=0), 'r')
# plt.loglog(freqs, np.mean(vib_off_psd.get_data(), axis=0), 'b')
# plt.legend(['Vibration on', 'Vibration off'])


# fwd = mne.make_forward_solution(
#     craw.info, 
#     trans='fsaverage', 
#     #src=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-vol-5-src.fif'), 
#     src=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'), 
#     bem=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'), 
#     eeg=True, 
#     mindist=5.0, 
#     n_jobs=16
# )

# noise_cov = mne.compute_raw_covariance(p_raws['FixRest'])

# inverse_operator = mne.minimum_norm.make_inverse_operator(p_raws['FixRest'].info, fwd, noise_cov)

# stc = mne.minimum_norm.apply_inverse_raw(p_raws['EcRest'], inverse_operator, lambda2=1.0 / 9.0, method='MNE')

# hstc = stc.crop(tmin=0, tmax=30).filter(l_freq=8, h_freq=13).apply_hilbert(envelope=True, n_jobs=16)


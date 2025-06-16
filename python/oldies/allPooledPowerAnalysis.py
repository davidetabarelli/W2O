import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


# Get all subjects data divided into all periods
p_raws = [];
all_events = []
for subject in subjects:
    craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
    p_raws.append(w2o.preliminary.extract_periods(craw, events, evt_dict, 0))    
    all_events.append(events)


# Define periods of interest
iperiods = ['FixRest', 'EcRest', 'Muscles', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution']


# Get all periods spectra, also as evoked structures
p_psds = []
#p_psd_evk = []
for s, subject in enumerate(subjects):
    psds = {ip : None for ip in iperiods}
    psds_evk = {ip : None for ip in iperiods}
    for period in iperiods: 
        
        if period == 'Muscles':
            lepochs = w2o.preliminary.extract_muscles_epochs(p_raws[s][period], all_events[s], evt_dict, 2.0, 0.5)
            psds[period] = lepochs.compute_psd(method='multitaper', fmin=2.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
        else:
            psds[period] = mne.make_fixed_length_epochs(p_raws[s][period], duration=2, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=2.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
        
        #info = mne.create_info(psds[period].info['ch_names'], 2.0, 'eeg')
        #info.set_montage('standard_1005')
        #psds_evk[period] = mne.EvokedArray(psds[period].get_data(), info)
        #psds_evk[period].comment  = period
        
    p_psds.append(psds)
    #p_psd_evk.append(psds_evk)
    
    del psds
    #del psds_evk




# Frequenze
freqs = p_psds[0]['FixRest'].freqs


# Medie di gruppo con canali esplosi
....####???

# Media sui canali per ciascun soggetto (matrice)
pld_p_psds = {ip : np.asarray([np.mean(p_psd[ip].get_data(), axis=0) for p_psd in p_psds]) for ip in iperiods}


# Grand averages with SEM
ga_pld_p_psds = {ip : np.mean(pld_p_psds[ip], axis=0) for ip in iperiods}
sem_pld_p_psds = {ip : np.std(pld_p_psds[ip], axis=0) / np.sqrt(N) for ip in iperiods}


# Grafico
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for ip in iperiods:
    ax.semilogy(freqs, ga_pld_p_psds[ip])
ax.legend(iperiods)
for ip in iperiods:
    plt.fill_between(freqs, ga_pld_p_psds[ip] - sem_pld_p_psds[ip], ga_pld_p_psds[ip] + sem_pld_p_psds[ip], alpha=0.1)
fig.set_size_inches([9,5])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('EEG Power')
fig.suptitle('Spettri dei periodi di interesse mediati sul gruppo con SEM.', fontsize=10)


# Spettri normalizzati sul FixRest, in dB
nperiod = 'FixRest'

n_pld_p_psds = {ip : 10.0 * np.log10(pld_p_psds[ip] / pld_p_psds[nperiod]) for ip in iperiods}

n_ga_p_psds = {ip : np.mean(n_pld_p_psds[ip], axis=0) for ip in iperiods}
n_sem_p_psds = {ip : np.std(n_pld_p_psds[ip], axis=0) / np.sqrt(N) for ip in iperiods}


# Grafico
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for ip in [ip for ip in iperiods if ip != nperiod]:
    ax.plot(freqs, n_ga_p_psds[ip])
ax.legend([ip for ip in iperiods if ip != nperiod])
for ip in [ip for ip in iperiods if ip != nperiod]:
    plt.fill_between(freqs, n_ga_p_psds[ip] - n_sem_p_psds[ip], n_ga_p_psds[ip] + n_sem_p_psds[ip], alpha=0.1)
fig.set_size_inches([9,5])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('EEG Power [dB]')
fig.suptitle('Spettri dei periodi di interesse normalizzati sul %s (in dB) e mediati sul gruppo con SEM.' % nperiod, fontsize=10)




## Bande di frequenza. Fai funzione che le estrae dal psd individuale. Poi ANOVA e post hoc.






##### OLDIES

# Compute all average power in frequency bands in periods of interests
iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'FixRest']
normPeriod = 'EcRest'
fbands = w2o.spectral.get_fbands_dict()
fb_df = pd.DataFrame({  
                        'Subject': pd.Series(dtype='str'), 
                        'Band': pd.Series(dtype='str'), 
                        'Period': pd.Series(dtype='str'), 
                        'Power': pd.Series(dtype='float'), 
                        'nPower': pd.Series(dtype='float'),     # Normalized
                    })

for s,subject in enumerate(subjects):
    
    for period in iperiods:
        
        psd = mne.make_fixed_length_epochs(p_raws[s][period], duration=2, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=2.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
   
        freqs = psd.freqs
        power = psd.get_data()
        del psd
    
        for band, fb in fbands.items():
            
            fb_idx = np.argwhere((freqs >= fb[0]) & (freqs <= fb[1])).reshape(-1)
            
            nrow ={}
            
            nrow['Subject'] = subject
            nrow['Band'] = band
            nrow['Period'] = period
            nrow['Power'] = np.mean(np.mean(np.sqrt(power), axis=0)[fb_idx]) # SQRT ????
            #nrow['Power'] = np.mean(np.mean(power, axis=0)[fb_idx])
            nrow['nPower'] = 0.0
            
            fb_df = pd.concat([fb_df, pd.DataFrame([nrow])], ignore_index=True)

for subject in subjects:
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


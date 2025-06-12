# Control factors

# 1) Vibration on Vibration off does not change spectra

import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


# Get subjects' vibration periods epoched data
all_epochs = []
vperiods = ['VibTest', 'Vib1', 'Vib2']
for subject in subjects:
    
    craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
    
    v_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 0, vperiods)
    
    epochs = mne.concatenate_epochs([ep for ep in [w2o.preliminary.extract_vib_epochs(v_raws[vp], events, evt_dict) for vp in v_raws.keys()] if ep != None])
    
    all_epochs.append(epochs)


# Compute subjects' vibration spectra, averaged and not
all_psds = {'VibOn': [], 'VibOff': []}
all_avg_psds = {'VibOn': [], 'VibOff': []}
for s, subject in enumerate(subjects):
    
    psd_on = all_epochs[s]['VibOn'].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg')
    psd_off = all_epochs[s]['VibOff'].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg')
        
    all_psds['VibOn'].append(psd_on)
    all_psds['VibOff'].append(psd_off)
    
    all_avg_psds['VibOn'].append(psd_on.average())
    all_avg_psds['VibOff'].append(psd_off.average())

freqs = all_psds['VibOn'][0].freqs    


# Average spectra on pooled electrodes
pld_psd = {}
pld_psd['VibOn'] = np.asarray([np.mean(all_avg_psds['VibOn'][s].get_data(), axis=0) for s in range(N)])
pld_psd['VibOff'] = np.asarray([np.mean(all_avg_psds['VibOff'][s].get_data(), axis=0) for s in range(N)])

# Group average with SEM
ga_pld_psd = {}
sem_pld_psd = {}

ga_pld_psd['VibOn'] = np.mean(pld_psd['VibOn'], axis=0)
sem_pld_psd['VibOn'] = np.std(pld_psd['VibOn'], axis=0) / np.sqrt(N)

ga_pld_psd['VibOff'] = np.mean(pld_psd['VibOff'], axis=0)
sem_pld_psd['VibOff'] = np.std(pld_psd['VibOff'], axis=0) / np.sqrt(N)



# Grafico
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.semilogy(freqs, ga_pld_psd['VibOn'], 'r')
ax.semilogy(freqs, ga_pld_psd['VibOff'], 'b')
ax.legend(ga_pld_psd.keys())
ax.fill_between(freqs, ga_pld_psd['VibOn'] - sem_pld_psd['VibOn'], ga_pld_psd['VibOn'] + sem_pld_psd['VibOn'], alpha=0.1, color='r')
ax.fill_between(freqs, ga_pld_psd['VibOff'] - sem_pld_psd['VibOff'], ga_pld_psd['VibOff'] + sem_pld_psd['VibOff'], alpha=0.1, color='r')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('EEG Power')
fig.suptitle('Spettri dei periodi di vibratore on e vibratore off mediati sul gruppo con SEM.', fontsize=10)



# Group statistics: two tailed paired t-test on spectra pooled on electrodes
thr = 0.05
tail = 0
M = 10000
stat_fun=mne.stats.ttest_1samp_no_p
X = pld_psd['VibOn'] - pld_psd['VibOff']
adj = mne.stats.combine_adjacency(len(freqs))

# plt.imshow(adj.toarray())

T, cl, clp, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
sig_cl = np.argwhere(clp <= thr).reshape(-1)


# Single subject statistics: two tailed paired t-test on spectra pooled on electrodes
all_cl = []
all_clp = []
all_sig_cl = []
all_lengths = []
for s, subject in enumerate(subjects):
    
    thr = 0.05
    tail = 0
    M = 10000
    stat_fun=mne.stats.ttest_1samp_no_p
    last = np.min([all_psds['VibOn'][s].shape[0], all_psds['VibOff'][s].shape[0]])
    X = np.mean(all_psds['VibOn'][s].get_data()[:last,:,:], axis=1) - np.mean(all_psds['VibOff'][s].get_data()[:last,:,:], axis=1)
    adj = mne.stats.combine_adjacency(len(freqs))
    T, cl, clp, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
    sig_cl = np.argwhere(clp <= thr).reshape(-1)
    
    all_cl.append(cl)
    all_clp.append(clp)
    all_sig_cl.append(sig_cl)
    all_lengths.append(last)


fig, axs = plt.subplots(5,2)
[ax.spines['top'].set_visible(False) for ax in axs.flat]
[ax.spines['right'].set_visible(False) for ax in axs.flat]
fig.set_size_inches([10,10])
i = 0
for s in np.argwhere([len(sc)!=0 for sc in all_sig_cl]).reshape(-1):
    
    sig_cl = all_sig_cl[s]
    cl = all_cl[s]
    clp = all_clp[s]
        
    axs.flat[i].semilogy(freqs, np.mean(all_psds['VibOn'][s].get_data(), axis=(0,1)), 'r--', linewidth=0.5)
    axs.flat[i].semilogy(freqs, np.mean(all_psds['VibOff'][s].get_data(), axis=(0,1)), 'b--', linewidth=0.5)
        
    for sc in sig_cl:
        axs.flat[i].semilogy(freqs[np.argwhere(cl[sc]).reshape(-1)], np.mean(all_psds['VibOn'][s].get_data(), axis=(0,1))[np.argwhere(cl[sc]).reshape(-1)], 'r', linewidth=2)
        axs.flat[i].semilogy(freqs[np.argwhere(cl[sc]).reshape(-1)], np.mean(all_psds['VibOff'][s].get_data(), axis=(0,1))[np.argwhere(cl[sc]).reshape(-1)], 'b', linewidth=2)
    
    axs.flat[i].set_xlabel('Frequency [Hz]', fontsize=7)
    axs.flat[i].set_ylabel('EEG Power', fontsize=7)
    
    axs.flat[i].text(0.4, 0.8, 'Subject %s (%d)- p-value = %s' % (subjects[s], all_lengths[s], str([clp[sc] for sc in sig_cl])), fontsize=8, va='center', ha='left', transform=axs.flat[i].transAxes)
    
    i = i + 1

fig.legend(['VibOn', 'VibOff'])

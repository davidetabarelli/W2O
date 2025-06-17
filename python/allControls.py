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


# Periods of interest
vperiods = ['VibTest', 'Vib1', 'Vib2']


# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Stat plots legendù
legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']


# Get subjects' spectra
psds = {'VibOn': [], 'VibOff': []}
avg_psds = {'VibOn': [], 'VibOff': []}
fb_psds = {'VibOn': {fb : [] for fb in fbands.keys()}, 'VibOff': {fb : [] for fb in fbands.keys()}}
for subject in subjects:
    
    lres = w2o.spectral.get_periods_psds(subject, vperiods, [])
    
    [psds[ip].append(lres[0][ip]) for ip in psds.keys()]
    [avg_psds[ip].append(lres[1][ip]) for ip in avg_psds.keys()]
    [fb_psds[ip][fb].append(lres[2][ip][fb]) for fb in fbands.keys() for ip in fb_psds.keys()]

    # Frequencies (all the same)
    freqs = lres[3]['VibOn']

# Compute electrode pooled (average) data
pld_avg_psds = {ip: [np.mean(avg_psds[ip][s].get_data(), axis=0) for s in range(N)] for ip in psds.keys()}
pld_fb_psds = {ip : {fb : [np.mean(fb_psds[ip][fb][s], axis=0) for s in range(N)] for fb in fbands.keys()} for ip in fb_psds.keys()}



# Compute all grand averages and SEMs
ga_avg_psds = {ip : np.mean(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0) for ip in avg_psds.keys()}
sem_avg_psds = {ip : np.std(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}

ga_pld_avg_psds = {ip : np.mean(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0) for ip in pld_avg_psds.keys()}
sem_pld_avg_psds = {ip : np.std(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}

ga_fb_psds = {ip : {fb: np.mean(np.asarray(fb_psds[ip][fb]), axis=0) for fb in fb_psds[ip].keys()} for ip in fb_psds.keys()}
sem_fb_psds = {ip : {fb: np.std(np.asarray(fb_psds[ip][fb]), axis=0)/np.sqrt(N) for fb in fb_psds[ip].keys()} for ip in fb_psds.keys()}

ga_pld_fb_psds = {ip : {fb: np.mean(np.asarray(pld_fb_psds[ip][fb]), axis=0) for fb in pld_fb_psds[ip].keys()} for ip in pld_fb_psds.keys()}
sem_pld_fb_psds = {ip : {fb: np.std(np.asarray(pld_fb_psds[ip][fb]), axis=0)/np.sqrt(N) for fb in pld_fb_psds[ip].keys()} for ip in pld_fb_psds.keys()}



# Group statistical analysis on pooled spectra - T-Test comparison between VibOn and VibOff
pld_avg_stat = w2o.statutils.pooled_spectra_1_samp_statistics([pld_avg_psds['VibOn'], pld_avg_psds['VibOff']])
fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds['VibOn'], ga_pld_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, pld_avg_stat['sig_cl'], pld_avg_stat['clp'], pld_avg_stat['cl'], pld_avg_stat['T'], legend)
fig.suptitle('Group analysys - Pooled electrodes')


# Group statistical analysis on spatially resolved spectra - T-Test comparison between VibOn and VibOff
avg_stat = w2o.statutils.spatial_spectra_1_samp_statistics([avg_psds['VibOn'], avg_psds['VibOff']])
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['VibOn'], ga_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['VibOn'][0].info, legend)
fig.suptitle('Group analysys - Spatially resolved')

# Group statistical analysis on frequency bands ... come? Spatial o no?

# Single subject statistical comparison, spatially resolved
all_subj_stats = []
for s in range(N):
    
    subject = subjects[s]
    
    stat_data = [[mne.time_frequency.SpectrumArray(psds[ip][s].get_data()[i,:,:], psds[ip][s].info, psds[ip][s].freqs) for i in range(psds[ip][s].get_data().shape[0])] for ip in psds.keys()]
    
    lstat = w2o.statutils.spatial_spectra_1_samp_statistics(stat_data)
    lsem = [np.std(np.mean(psds[ip][s].get_data(), axis=1), axis=0) for ip in psds.keys()]
    nE = np.max([len(psds['VibOn'][s]), len(psds['VibOff'][s])])
    
    all_subj_stats.append(lstat)
    
    if len(lstat['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_power_cluster_summary([avg_psds['VibOn'][s].get_data(), avg_psds['VibOff'][s].get_data()], lsem, freqs, lstat['sig_cl'], lstat['clp'], lstat['cl'], lstat['T'], psds['VibOn'][s][0].info, legend)
        fig.suptitle('Subject %s (%d epochs)' % (subject, nE))
    
    










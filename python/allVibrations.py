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
vperiods = ['VibTest', 'Vib1', 'Vib2', 'FixRest']
norm_period = 'FixRest'


# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Stat plots legendù
legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']


# Get subjects' spectra
psds = {'VibOn': [], 'VibOff': []}
avg_psds = {'VibOn': [], 'VibOff': []}
fb_psds = {'VibOn': {fb : [] for fb in fbands.keys()}, 'VibOff': {fb : [] for fb in fbands.keys()}}
for subject in subjects:
    
    lres = w2o.spectral.get_periods_psds(subject, vperiods, norm_period)
    
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
pld_stat = w2o.statistics.pooled_spectra_1_samp_t_test([pld_avg_psds['VibOn'], pld_avg_psds['VibOff']])
pld_fig, pld_axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds['VibOn'], ga_pld_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, pld_stat['sig_cl'], pld_stat['clp'], pld_stat['cl'], pld_stat['T'], legend)
pld_fig.suptitle('Group analisys - Pooled electrodes')


# Group statistical analysis on spatially resolved spectra - T-Test comparison between VibOn and VibOff
avg_stat = w2o.statistics.spatial_spectra_1_samp_t_test([avg_psds['VibOn'], avg_psds['VibOff']])
avg_fig, avg_axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['VibOn'], ga_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['VibOn'][0].info, legend)
avg_fig.suptitle('Group analisys - Spatially resolved') 

# Group statistical analysis on frequency bands, spatially resolved.
fb_figs = []
fb_axs = []
fb_stat = {}
for fb in fbands.keys():
    fb_stat[fb] = w2o.statistics.fbands_spectra_1_samp_t_test([fb_psds['VibOn'][fb], fb_psds['VibOff'][fb]], avg_psds['VibOn'][0].info)    
    if len(fb_stat[fb]['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds['VibOn'][fb], fb_psds['VibOff'][fb]], fb_stat[fb]['sig_cl'], fb_stat[fb]['clp'], fb_stat[fb]['cl'], fb_stat[fb]['T'], avg_psds['VibOn'][0].info, conditions=['VibOn', 'VibOff'])
        fig.suptitle('%s (%.0f - %.0f Hz)' % (fb, fbands[fb][0], fbands[fb][1]))
        fb_figs.append(fig)
        fb_axs.append(axs)


# Single subject statistical comparison, spatially resolved
subs_figs = []
all_subj_stats = []
for s in range(N):
    
    subject = subjects[s]
    
    nE = np.min([psds['VibOn'][s].shape[0], psds['VibOff'][s].shape[0]])
    
    stat_data = [[mne.time_frequency.SpectrumArray(psds[ip][s].get_data()[i,:,:], psds[ip][s].info, psds[ip][s].freqs) for i in range(nE)] for ip in psds.keys()]
    
    lstat = w2o.statistics.spatial_spectra_1_samp_t_test(stat_data)
    lsem = [np.std(np.mean(psds[ip][s].get_data()[:nE,:,:], axis=1), axis=0) for ip in psds.keys()]
        
    all_subj_stats.append(lstat)
    
    if len(lstat['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_power_cluster_summary([avg_psds['VibOn'][s].get_data(), avg_psds['VibOff'][s].get_data()], lsem, freqs, lstat['sig_cl'], lstat['clp'], lstat['cl'], lstat['T'], psds['VibOn'][0].info, legend)
        fig.suptitle('Subject %s - Epochs %d' % (subject, nE))
        subs_figs.append(fig)
    else:
        subs_figs.append(None)







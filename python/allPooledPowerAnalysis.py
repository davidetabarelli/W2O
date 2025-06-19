import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


# Periods of interest
iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'FixRest']
norm_period = 'FixRest'
#norm_period = []

# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Stat plots legend
#legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']
legend=[]


# Get subjects' spectra
psds = {ip : [] for ip in iperiods}
avg_psds = {ip : [] for ip in iperiods}
fb_psds = {ip: {fb : [] for fb in fbands.keys()} for ip in iperiods}
for subject in subjects:
    
    lres = w2o.spectral.get_periods_psds(subject, iperiods, norm_period)
    
    [psds[ip].append(lres[0][ip]) for ip in psds.keys()]
    [avg_psds[ip].append(lres[1][ip]) for ip in avg_psds.keys()]
    [fb_psds[ip][fb].append(lres[2][ip][fb]) for fb in fbands.keys() for ip in fb_psds.keys()]

    # Frequencies (all the same)
    freqs = lres[3][iperiods[0]]
    
    
    
# Epochs number
n_epochs = {}
for ip in iperiods:
    n_epochs[ip] = np.asarray([psds[ip][s].shape[0] for s in range(N)])
    
[[np.min(ne), np.max(ne), np.mean(ne)] for ne in n_epochs.values()]
    

# Compute electrode pooled (average) data
pld_avg_psds = {ip: [np.mean(avg_psds[ip][s].get_data(), axis=0) for s in range(N)] for ip in psds.keys()}
pld_fb_psds = {ip : {fb : [np.mean(fb_psds[ip][fb][s], axis=0) for s in range(N)] for fb in fbands.keys()} for ip in fb_psds.keys()}


# Compute all grand averages and SEMs for non frequency bands spectra
ga_avg_psds = {ip : np.mean(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0) for ip in avg_psds.keys()}
sem_avg_psds = {ip : np.std(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}

ga_pld_avg_psds = {ip : np.mean(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0) for ip in pld_avg_psds.keys()}
sem_pld_avg_psds = {ip : np.std(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}

ga_fb_psds = {ip : {fb: np.mean(np.asarray(fb_psds[ip][fb]), axis=0) for fb in fb_psds[ip].keys()} for ip in fb_psds.keys()}
sem_fb_psds = {ip : {fb: np.std(np.asarray(fb_psds[ip][fb]), axis=0)/np.sqrt(N) for fb in fb_psds[ip].keys()} for ip in fb_psds.keys()}

ga_pld_fb_psds = {ip : {fb: np.mean(np.asarray(pld_fb_psds[ip][fb]), axis=0) for fb in pld_fb_psds[ip].keys()} for ip in pld_fb_psds.keys()}
sem_pld_fb_psds = {ip : {fb: np.std(np.asarray(pld_fb_psds[ip][fb]), axis=0)/np.sqrt(N) for fb in pld_fb_psds[ip].keys()} for ip in pld_fb_psds.keys()}


# Periods for statistical tests
if norm_period == []:
    stat_periods = iperiods
else:
    stat_periods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution']

# Info structure
info = avg_psds['Orgasm'][0].info


# Simple spectra plot
if norm_period == []:
    fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in iperiods], [sem_pld_avg_psds[k] for k in iperiods], freqs, [], [], [], [], iperiods)
else:
    fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, [], [], [], [], stat_periods)
[line.set_linewidth(1.5) for line in axs['PLOT'].lines]
[line.set_linestyle('-') for line in axs['PLOT'].lines]
axs['PLOT'].legend(stat_periods)
if norm_period == []:
    fig.suptitle('Non normalized power spectra')
else:
    fig.suptitle('%s normalized power spectra' % norm_period)



# Pooled electrodes ANOVA
pld_F_stat = w2o.statistics.pooled_spectra_1w_rm_ANOVA([pld_avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, pld_F_stat['sig_cl'], pld_F_stat['clp'], pld_F_stat['cl'], pld_F_stat['F'], stat_periods)

# Spatially resolved ANOVA
F_stat = w2o.statistics.spatial_spectra_1w_rm_ANOVA([avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, F_stat['sig_cl'], F_stat['clp'], F_stat['cl'], F_stat['F'], info, stat_periods)

# Frequency bands spatially resolved ANOVA
fb_F_stat = {}
for fb in fbands.keys():
    fb_F_stat[fb] = w2o.statistics.fbands_spectra_1w_rm_ANOVA([fb_psds[sp][fb] for sp in stat_periods], info)
    if len(fb_F_stat[fb]['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds[sp][fb] for sp in stat_periods], fb_F_stat[fb]['sig_cl'], fb_F_stat[fb]['clp'], fb_F_stat[fb]['cl'], fb_F_stat[fb]['F'], info, conditions=stat_periods)
        fig.suptitle('%s (%.0f - %.0f Hz)' % (fb, fbands[fb][0], fbands[fb][1]))



# Post hoc
ph_combs = list(itertools.combinations(stat_periods,2))


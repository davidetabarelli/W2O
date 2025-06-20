# Muscolar activity test

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
iperiods = ['EcRest', 'Muscles', 'Orgasm', 'FixRest']
norm_period = 'FixRest'

# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Stat plots legend
#legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']
legend=['EcRest', 'Muscles', 'Orgasm']


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


# Periods for F-tests
stat_periods = ['EcRest', 'Muscles', 'Orgasm']

# Info structure
info = avg_psds['Orgasm'][0].info


# Pooled electrodes ANOVA
pld_F_stat = w2o.statistics.pooled_spectra_1w_rm_ANOVA([pld_avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, pld_F_stat['sig_cl'], pld_F_stat['clp'], pld_F_stat['cl'], pld_F_stat['F'], stat_periods)
fig.suptitle('Pooled electrodes ANOVA')

# Spatially resolved ANOVA
F_stat = w2o.statistics.spatial_spectra_1w_rm_ANOVA([avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, F_stat['sig_cl'], F_stat['clp'], F_stat['cl'], F_stat['F'], info, stat_periods)
fig.suptitle('Spatially resolved ANOVA')

# Frequency bands spatially resolved ANOVA
fb_F_stat = {}
for fb in fbands.keys():
    fb_F_stat[fb] = w2o.statistics.fbands_spectra_1w_rm_ANOVA([fb_psds[sp][fb] for sp in stat_periods], info)
    if len(fb_F_stat[fb]['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds[sp][fb] for sp in stat_periods], fb_F_stat[fb]['sig_cl'], fb_F_stat[fb]['clp'], fb_F_stat[fb]['cl'], fb_F_stat[fb]['F'], info, conditions=stat_periods)
        fig.suptitle('ANOVA - %s (%.0f - %.0f Hz)' % (fb, fbands[fb][0], fbands[fb][1]))


# Post hoc
ph_combs = list(itertools.combinations(stat_periods,2))
ph_combs = [(pc[1],pc[0]) for pc in ph_combs]  # Invert order

# Pooled electrodes
pld_F_stat['post_hoc'] = {}
for pc in ph_combs:    
    lstat = w2o.statistics.pooled_spectra_1_samp_t_test([pld_avg_psds[k] for k in pc])
    pld_F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])] = lstat
    if len(lstat['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in pc], [sem_pld_avg_psds[k] for k in pc], freqs, lstat['sig_cl'], lstat['clp'], lstat['cl'], lstat['T'], pc)
        fig.suptitle('Pooled electrodes %s vs %s T-Test' % (pc[0], pc[1]))


# Spatially resolved
F_stat['post_hoc'] = {}
for pc in ph_combs:    
    lstat = w2o.statistics.spatial_spectra_1_samp_t_test([avg_psds[k] for k in pc])
    F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])] = lstat
    if len(lstat['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds[k] for k in pc], [sem_pld_avg_psds[k] for k in pc], freqs, lstat['sig_cl'], lstat['clp'], lstat['cl'], lstat['T'], info, pc)
        fig.suptitle('Spatially resolved %s vs %s T-Test' % (pc[0], pc[1]))


# Frequency bands spatially resolved
for fb in fbands.keys():
    fb_F_stat[fb]['post_hoc'] = {}
    for pc in ph_combs:
        lstat = w2o.statistics.fbands_spectra_1_samp_t_test([fb_psds[sp][fb] for sp in pc], info)
        fb_F_stat[fb]['post_hoc']['%s_%s' % (pc[0], pc[1])] = lstat
        if len(lstat['sig_cl']) > 0:
            fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds[sp][fb] for sp in pc], lstat['sig_cl'], lstat['clp'], lstat['cl'], lstat['T'], info, conditions=pc)
            fig.suptitle('%s (%.0f - %.0f Hz) - %s vs %s T-Test' % (fb, fbands[fb][0], fbands[fb][1], pc[0], pc[1]))


# Save all results
##### Big res dictionary ...











        





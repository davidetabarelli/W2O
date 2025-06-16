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


# Compute all grand averages and SEMs for non frequency bands spectra
ga_avg_psds = {ip : np.mean(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0) for ip in avg_psds.keys()}
sem_avg_psds = {ip : np.std(np.asarray([avg_psds[ip][s].get_data() for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}

ga_pld_avg_psds = {ip : np.mean(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0) for ip in pld_avg_psds.keys()}
sem_pld_avg_psds = {ip : np.std(np.asarray([pld_avg_psds[ip][s] for s in range(N)]), axis=0)/np.sqrt(N) for ip in avg_psds.keys()}




# Group statistical analysis on pooled spectra - T-Test comparison between VibOn and VibOff
pld_avg_stat = w2o.statutils.pooled_spectra_1_samp_statistics([pld_avg_psds['VibOn'], pld_avg_psds['VibOff']])

# Plot results
w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds['VibOn'], ga_pld_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, pld_avg_stat['sig_cl'], pld_avg_stat['clp'], pld_avg_stat['cl'], pld_avg_stat['T'], legend)


# Group statistical analysis on spatially resolved spectra - T-Test comparison between VibOn and VibOff
avg_stat = w2o.statutils.spatial_spectra_1_samp_statistics([avg_psds['VibOn'], avg_psds['VibOff']])

# Plot results
w2o.viz.plot_power_cluster_summary([ga_avg_psds['VibOn'], ga_avg_psds['VibOff']], [sem_pld_avg_psds['VibOn'], sem_pld_avg_psds['VibOff']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['VibOn'][0].info, legend)















# Get subjects' vibration periods epoched data
all_epochs = []
vperiods = ['VibTest', 'Vib1', 'Vib2']
for subject in subjects:
    
    craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
    
    v_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 0, vperiods)
    
    epochs = mne.concatenate_epochs([ep for ep in [w2o.preliminary.extract_vib_epochs(v_raws[vp], events, evt_dict) for vp in v_raws.keys()] if ep != None])
    
    all_epochs.append(epochs)




# Compute subjects' vibration spectra, averaged and not

all_avg_psds = {'VibOn': [], 'VibOff': []}
all_freqs = []
for s, subject in enumerate(subjects):
    
    psd_on = all_epochs[s]['VibOn'].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg')
    psd_off = all_epochs[s]['VibOff'].compute_psd(method='multitaper', fmin=1.0, fmax=45, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg')
    
    assert np.all(psd_on.freqs == psd_off.freqs)
    
    all_freqs.append(psd_on.freqs)
    
    all_psds['VibOn'].append(psd_on)
    all_psds['VibOff'].append(psd_off)
    
    all_avg_psds['VibOn'].append(psd_on.average())
    all_avg_psds['VibOff'].append(psd_off.average())

assert(np.max(np.mean(np.asarray(all_freqs), axis=0) - all_freqs[0]) <= 1e-10)
freqs = psd_on.freqs
del all_freqs




# Average spectra on pooled electrodes, computing group average and SEM
pld_psd = {}
pld_psd['VibOn'] = np.asarray([np.mean(all_avg_psds['VibOn'][s].get_data(), axis=0) for s in range(N)])
pld_psd['VibOff'] = np.asarray([np.mean(all_avg_psds['VibOff'][s].get_data(), axis=0) for s in range(N)])
ga_pld_psd = {}
sem_pld_psd = {}
ga_pld_psd['VibOn'] = np.mean(pld_psd['VibOn'], axis=0)
sem_pld_psd['VibOn'] = np.std(pld_psd['VibOn'], axis=0) / np.sqrt(N)
ga_pld_psd['VibOff'] = np.mean(pld_psd['VibOff'], axis=0)
sem_pld_psd['VibOff'] = np.std(pld_psd['VibOff'], axis=0) / np.sqrt(N)




# Group statistics: two tailed paired t-test on spectra pooled on electrodes
pval = 0.05
thr = sp.stats.t.ppf(1 - pval / 2, N-1)
tail = 0
M = 10000
X = pld_psd['VibOn'] - pld_psd['VibOff']
adj = mne.stats.combine_adjacency(len(freqs))

T, cl, clp, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
sig_cl = np.argwhere(clp <= pval).reshape(-1)
print('Averaged electrodes group analysis: significant clusters = %s'% str(sig_cl))

if len(sig_cl) != 0:
    legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']    
    fig, axs = w2o.viz.plot_pooled_power_cluster_summary(   [ga_pld_psd['VibOn'], ga_pld_psd['VibOff']],
                                                            [sem_pld_psd['VibOn'], sem_pld_psd['VibOff']],
                                                            freqs, sig_cl, clp, cl, T, legend
                                                        )






# Single subject statistics: two tailed paired t-test on spectra pooled on electrodes
all_cl = []
all_clp = []
all_sig_cl = []
all_N = []
sig_figs = []
for s, subject in enumerate(subjects):
    
    lN = np.min([all_psds['VibOn'][s].shape[0], all_psds['VibOff'][s].shape[0]])
    pval = 0.05
    thr = sp.stats.t.ppf(1 - pval / 2, lN-1)
    tail = 0
    M = 10000
    
    X = np.mean(all_psds['VibOn'][s].get_data()[:lN,:,:], axis=1) - np.mean(all_psds['VibOff'][s].get_data()[:lN,:,:], axis=1)
    adj = mne.stats.combine_adjacency(len(freqs))
    T, cl, clp, H0 = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
    sig_cl = np.argwhere(clp <= pval).reshape(-1)
    
    all_cl.append(cl)
    all_clp.append(clp)
    all_sig_cl.append(sig_cl)
    all_N.append(lN)

    if len(sig_cl) != 0:
        legend = ['Vib On (all)', 'Vib Off (all)', 'Vib On (cluster)', 'VibOff (cluster)']    
        fig, axs = w2o.viz.plot_pooled_power_cluster_summary(    [np.mean(all_psds['VibOn'][s].get_data()[:lN,:,:], axis=(0,1)), np.mean(all_psds['VibOff'][s].get_data()[:lN,:,:], axis=(0,1))],
                                                                 [],
                                                                 freqs, sig_cl, clp, cl, T, legend
                                                        )
        fig.suptitle('Subject %s' % subject)
        sig_figs.append(fig)





# Group statistics: spatio spectral two tailed paired t-test
pval = 0.05
thr = sp.stats.t.ppf(1 - pval / 2, N-1)
tail = 0
M = 10000
stat_fun=mne.stats.ttest_1samp_no_p
X = np.transpose(np.asarray([all_avg_psds['VibOn'][s].get_data() - all_avg_psds['VibOff'][s].get_data() for s in range(N)]), (0,2,1))
sadj, ch_names = mne.channels.find_ch_adjacency(craw.copy().pick('eeg').info, 'eeg')
adj = mne.stats.combine_adjacency(len(freqs), sadj)

T, cl, clp, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
sig_cl = np.argwhere(clp <= pval).reshape(-1)
print('Spatial group analysis: significant clusters = %s'% str(sig_cl))
info = all_epochs[s]['VibOn'].copy().pick('eeg').info
if len(sig_cl) != 0:
    fig, axs = w2o.viz.plot_power_cluster_summary(  [np.mean(np.asarray([all_avg_psds['VibOn'][s].get_data() for s in range(N)]), axis=0), np.mean(np.asarray([all_avg_psds['VibOff'][s].get_data() for s in range(N)]), axis=0)],
                                                  freqs, sig_cl, clp, cl, T, info, legend)







# Single subject statistics: two tailed paired t-test
all_T = []
all_cl = []
all_clp = []
all_sig_cl = []
all_N = []
sig_figs = []
for s, subject in enumerate(subjects):
    
    lN = np.min([all_psds['VibOn'][s].shape[0], all_psds['VibOff'][s].shape[0]])
    
    pval = 0.05
    thr = sp.stats.t.ppf(1 - pval / 2, lN-1)
    tail = 0
    M = 10000
    
    info = all_epochs[s]['VibOn'].copy().pick('eeg').info
    X = np.transpose(all_psds['VibOn'][s].get_data()[:lN,:,:] - all_psds['VibOff'][s].get_data()[:lN,:,:], (0,2,1))
    sadj, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')
    adj = mne.stats.combine_adjacency(len(freqs), sadj)
    
    T, cl, clp, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X, threshold=thr, n_permutations=M, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask')
    sig_cl = np.argwhere(clp <= pval).reshape(-1)
    
    all_T.append(T)
    all_cl.append(cl)
    all_clp.append(clp)
    all_sig_cl.append(sig_cl)
    all_N.append(lN)
    
    if len(sig_cl) != 0:
        fig, axs = w2o.viz.plot_power_cluster_summary(  [np.mean(all_psds['VibOn'][s].get_data()[:lN,:,:], axis=0), np.mean(all_psds['VibOff'][s].get_data()[:lN,:,:], axis=0)],
                                                      freqs, sig_cl, clp, cl, T, info, legend)
        fig.suptitle('Subject %s' % subject)
        sig_figs.append(fig)




# Frequency bands with/without pooled electrodes


# Group statistics (pooled or not)


# Single subject statistics (pooled or not)

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
iperiods = ['FixRest', 'EcRest', 'Muscles', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution']
norm_period = 'FixRest'

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
stat_periods = ['EcRest', 'Muscles', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution']
#stat_periods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm']

# Info structure
info = avg_psds['Orgasm'][0].info



# Pooled electrodes F-Test
pld_F_stat = w2o.statistics.pooled_spectra_1w_rm_ANOVA([pld_avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_pooled_power_cluster_summary([ga_pld_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, pld_F_stat['sig_cl'], pld_F_stat['clp'], pld_F_stat['cl'], pld_F_stat['F'], stat_periods)

# Spatially resolved F-Test
F_stat = w2o.statistics.spatial_spectra_1w_rm_ANOVA([avg_psds[k] for k in stat_periods])
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds[k] for k in stat_periods], [sem_pld_avg_psds[k] for k in stat_periods], freqs, F_stat['sig_cl'], F_stat['clp'], F_stat['cl'], F_stat['F'], info, stat_periods)

# Frequency bands spatially resolved F-Test
fb_F_stat = {}
for fb in fbands.keys():
    fb_F_stat[fb] = w2o.statistics.fbands_spectra_1w_rm_ANOVA([fb_psds[sp][fb] for sp in stat_periods], info)
    if len(fb_F_stat[fb]['sig_cl']) > 0:
        fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds[sp][fb] for sp in stat_periods], fb_F_stat[fb]['sig_cl'], fb_F_stat[fb]['clp'], fb_F_stat[fb]['cl'], fb_F_stat[fb]['F'], info, conditions=stat_periods)
        fig.suptitle('%s (%.0f - %.0f Hz)' % (fb, fbands[fb][0], fbands[fb][1]))

###### Preliminary ...

avg_stat = w2o.statistics.spatial_spectra_1_samp_t_test([avg_psds['Orgasm'], avg_psds['EcRest']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Orgasm'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Orgasm'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Orgasm'][0].info, ['Orgasm', 'Eyes closed Rest'])
fig.suptitle('Orgasm VS Eyes Closed rest - Group statistics')

avg_stat = w2o.statistics.spatial_spectra_1_samp_t_test([avg_psds['Orgasm'], avg_psds['Muscles']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Orgasm'], ga_avg_psds['Muscles']], [sem_pld_avg_psds['Orgasm'], sem_pld_avg_psds['Muscles']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Orgasm'][0].info, ['Orgasm', 'Muscles'])
fig.suptitle('Orgasm VS Muscles - Group statistics')

avg_stat = w2o.statistics.spatial_spectra_1_samp_t_test([avg_psds['Muscles'], avg_psds['EcRest']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Muscles'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Muscles'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Muscles'][0].info, ['Muscles', 'Eyes closed Rest'])
fig.suptitle('Muscles VS Eyes Closed rest - Group statistics')


avg_stat = w2o.statistics.spatial_spectra_1_samp_statistics([avg_psds['Masturbation'], avg_psds['EcRest']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Masturbation'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Masturbation'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Masturbation'][0].info, ['Masturbation', 'Eyes closed Rest'])
fig.suptitle('Masturbation VS Eyes Closed rest - Group statistics')

avg_stat = w2o.statistics.spatial_spectra_1_samp_statistics([avg_psds['Pleateau'], avg_psds['EcRest']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Pleateau'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Pleateau'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Pleateau'][0].info, ['Pleateau', 'Eyes closed Rest'])
fig.suptitle('Pleateau VS Eyes Closed rest - Group statistics')

avg_stat = w2o.statistics.spatial_spectra_1_samp_statistics([avg_psds['Resolution'], avg_psds['EcRest']], 0.01)
fig, axs = w2o.viz.plot_power_cluster_summary([ga_avg_psds['Resolution'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Resolution'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Resolution'][0].info, ['Resolution', 'Eyes closed Rest'])
fig.suptitle('Resolution VS Eyes Closed rest - Group statistics')



info = avg_psds['EcRest'][0].info
fband = 'Beta'
avg_stat = w2o.statistics.fbands_spectra_F_statistics([fb_psds['EcRest'][fband], fb_psds['Orgasm'][fband], fb_psds['Muscles'][fband], fb_psds['Masturbation'][fband]], info)



avg_stat = w2o.statistics.spatial_spectra_F_statistics([avg_psds['EcRest'], avg_psds['Masturbation'], avg_psds['Pleateau'], avg_psds['Orgasm'], avg_psds['Muscles'], avg_psds['Resolution']])
fig, axs = w2o.viz.plot_power_cluster_summary(  [ga_avg_psds['EcRest'], ga_avg_psds['Masturbation'], ga_avg_psds['Pleateau'], ga_avg_psds['Orgasm'], ga_avg_psds['Muscles'], ga_avg_psds['Resolution']], 
                                                [sem_pld_avg_psds['EcRest'], sem_pld_avg_psds['Masturbation'], sem_pld_avg_psds['Pleateau'], sem_pld_avg_psds['Orgasm'], sem_pld_avg_psds['Muscles'], sem_pld_avg_psds['Resolution']], 
                                                freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['F'], avg_psds['EcRest'][0].info, 
                                                ['EcRest', 'Masturbation', 'Pleauteau', 'Orgasm', 'Muscles', 'Resolution'])

avg_stat = w2o.statistics.spatial_spectra_F_statistics([avg_psds['Masturbation'], avg_psds['Pleateau'], avg_psds['Orgasm'], avg_psds['Resolution']])
fig, axs = w2o.viz.plot_power_cluster_summary(  [ga_avg_psds['Masturbation'], ga_avg_psds['Pleateau'], ga_avg_psds['Orgasm'], ga_avg_psds['Resolution']], 
                                                [sem_pld_avg_psds['Masturbation'], sem_pld_avg_psds['Pleateau'], sem_pld_avg_psds['Orgasm'], sem_pld_avg_psds['Resolution']], 
                                                freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['F'], avg_psds['EcRest'][0].info, 
                                                ['Masturbation', 'Pleauteau', 'Orgasm', 'Resolution'])

fb_stat = {}
for fb in fbands.keys():
    
    fb_stat[fb] = w2o.statistics.fbands_spectra_1_samp_statistics([fb_psds['EcRest'][fb], fb_psds['Masturbation'][fb], fb_psds['Pleateau'][fb], fb_psds['Orgasm'][fb]], avg_psds['EcRest'][0].info)
    
    fig, axs = w2o.viz.plot_fbands_power_cluster_summary([fb_psds['EcRest'][fb], fb_psds['Masturbation'][fb], fb_psds['Pleateau'][fb], fb_psds['Orgasm'][fb]], fb_stat[fb]['sig_cl'], fb_stat[fb]['clp'], fb_stat[fb]['cl'], fb_stat[fb]['T'], avg_psds['EcRest'][0].info, conditions=['EcRest', 'Masturbation', 'Pleateau', 'Orgasm'])
    fig.suptitle('%s' % fb)


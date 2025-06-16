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



######

avg_stat = w2o.statutils.spatial_spectra_1_samp_statistics([avg_psds['Orgasm'], avg_psds['EcRest']])
w2o.viz.plot_power_cluster_summary([ga_avg_psds['Orgasm'], ga_avg_psds['EcRest']], [sem_pld_avg_psds['Orgasm'], sem_pld_avg_psds['EcRest']], freqs, avg_stat['sig_cl'], avg_stat['clp'], avg_stat['cl'], avg_stat['T'], avg_psds['Orgasm'][0].info, legend)

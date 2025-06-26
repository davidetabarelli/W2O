import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import fabric
import paramiko
import joblib

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'FixRest']
norm_period = 'FixRest'

# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Source reconstruction parameters
method = 'dSPM'
snr = 1  # Evoked (average) activity intversion
lambda2 = 1/snr**2

# Atlas
labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub')[:448]


# All structures ... TODO DECIDE
all_p_lb_spds = {ip : [] for ip in iperiods}
all_avg_p_lb_spds = {ip : [] for ip in iperiods}
all_freqs = {ip : [] for ip in iperiods}
for subject in subjects:
    
    p_lb_spds, avg_p_lb_spds, freqs, labels = w2o.sources.get_periods_source_psds(subject, iperiods, norm_period=norm_period, method=method, cov_period='FixRest')
    
    [all_p_lb_spds[ip].append(p_lb_spds[ip]) for ip in all_p_lb_spds.keys()]
    [all_avg_p_lb_spds[ip].append(avg_p_lb_spds[ip]) for ip in all_avg_p_lb_spds.keys()]
    [all_freqs[ip].append(freqs[ip]) for ip in all_freqs.keys()]

freqs = all_freqs[norm_period][0]


# Grand averages and SEMs
ga_avg_lb_psds = {ip : np.mean(np.asarray(all_avg_p_lb_spds[ip]), axis=0) for ip in all_avg_p_lb_spds.keys()}
sem_avg_lb_psds = {ip : np.std(np.asarray(all_avg_p_lb_spds[ip]), axis=0) / np.sqrt(N) for ip in all_avg_p_lb_spds.keys()}

# Source projections
stc_ga_avg_lb_psds = {ip : mne.labels_to_stc(labels, ga_avg_lb_psds[ip], tmin=freqs[0], tstep=np.diff(freqs[:2]), subject='fsaverage') for ip in iperiods}


# Statistics (ANOVA)
spectra = [all_avg_p_lb_spds[ip] for ip in iperiods[:-1]]
alpha = 0.05
permutations = w2o.statistics.get_permutation_number()

# Prepare stat data
X = [np.transpose(np.asarray([sp for sp in spectra[i]]), (0,2,1)) for i in range(len(spectra))]

# Get sample size
N = len(X[0])

# Conditions
C = len(X)

# Degrees
#dfn = C - 1
#dfd = N - C

# Adjacency
src = mne.read_source_spaces(os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
ladj = w2o.sources.get_labels_adjacency(src, labels, 'aparc_sub')
adj = mne.stats.combine_adjacency(len(freqs), ladj)

# One-way repeated-measures ANOVA
def stat_fun(*args):
    # get f-values only.
    return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[C], effects='A', return_pvals=False)[0]

# Statistical threshold
thr = mne.stats.f_threshold_mway_rm(N, [C], 'A', alpha)

F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=stat_fun, adjacency=adj, n_jobs=w2o.utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)

sig_cl = np.argwhere(clp <= alpha).reshape(-1)

res = {'F': F, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}







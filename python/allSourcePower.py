import numpy as np
import itertools

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'FixRest']
norm_period = 'FixRest'
stat_periods = iperiods[:-1]

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
all_fb_psds = {ip: {fb : [] for fb in fbands.keys()} for ip in iperiods}
all_freqs = {ip : [] for ip in iperiods}
for subject in subjects:
    
    p_lb_spds, avg_p_lb_spds, fb_p_lb_psds, freqs, labels = w2o.sources.get_periods_source_psds(subject, iperiods, norm_period=norm_period, method=method, cov_period='FixRest')
    
    [all_p_lb_spds[ip].append(p_lb_spds[ip]) for ip in all_p_lb_spds.keys()]
    [all_avg_p_lb_spds[ip].append(avg_p_lb_spds[ip]) for ip in all_avg_p_lb_spds.keys()]
    [all_freqs[ip].append(freqs[ip]) for ip in all_freqs.keys()]

freqs = all_freqs[norm_period][0]


# Grand averages and SEMs
ga_avg_lb_psds = {ip : np.mean(np.asarray(all_avg_p_lb_spds[ip]), axis=0) for ip in all_avg_p_lb_spds.keys()}
sem_avg_lb_psds = {ip : np.std(np.asarray(all_avg_p_lb_spds[ip]), axis=0) / np.sqrt(N) for ip in all_avg_p_lb_spds.keys()}

# Source projections
stc_ga_avg_lb_psds = {ip : mne.labels_to_stc(labels, ga_avg_lb_psds[ip], tmin=freqs[0], tstep=np.diff(freqs[:2]), subject='fsaverage') for ip in iperiods}


# Periods for statistical tests
if norm_period == []:
    stat_periods = iperiods
else:
    stat_periods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution']

import joblib
F_stat = joblib.load('F_stat.joblib')

# Spatial ANOVA
F_stat = w2o.statistics.labels_spectra_1w_rm_ANOVA([all_avg_p_lb_spds[ip] for ip in stat_periods], 'aparc_sub')
w2o.viz.plot_labels_power_cluster_summary([ga_avg_lb_psds[sp] for sp in stat_periods], [sem_avg_lb_psds[sp] for sp in stat_periods], freqs, F_stat['sig_cl'], F_stat['clp'], F_stat['cl'], F_stat['F'], labels, stat_periods)
  


# Frequency bands ANOVA
# TODO


# Post hocs
ph_combs = list(itertools.combinations(stat_periods,2))
ph_combs = [(pc[1],pc[0]) for pc in ph_combs]  # Invert order
ph_combs[-1] = ph_combs[-1][::-1]
ph_combs[-2] = ph_combs[-2][::-1]

# Spatial
F_stat['post_hoc'] = {}
for pc in ph_combs[7:]:    
    lstat = w2o.statistics.labels_spectra_1_samp_t_test([all_avg_p_lb_spds[ip] for ip in pc], 'aparc_sub', alpha=0.01, tail=0)
    F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])] = lstat


w2o.viz.plot_labels_power_cluster_summary([ga_avg_lb_psds[sp] for sp in pc], [sem_avg_lb_psds[sp] for sp in pc], freqs, F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])]['sig_cl'], F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])]['clp'], F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])]['cl'], F_stat['post_hoc']['%s_%s' % (pc[0], pc[1])]['T'], labels, pc)


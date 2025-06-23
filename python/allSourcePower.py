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


iperiods = ['EcRest', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'FixRest']
norm_period = 'FixRest'

# Frequency bands
fbands = w2o.spectral.get_fbands_dict()

# Source reconstruction parameters
method = 'eLORETA'
snr = 1  # Evoked (average) activity intversion
lambda2 = 1/snr**2

s = 0
subject = subjects[s]

craw, events, evt_dict = w2o.preliminary.get_clean_data(subjects[0])
craw.pick('eeg')
p_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 0, iperiods)

p_epochs = {}
for ip in p_raws.keys():
    lepochs = mne.make_fixed_length_epochs(p_raws[ip], duration=2, proj=True, reject_by_annotation=True, overlap=0.5)
    lepochs.drop_bad()
    p_epochs[ip] = lepochs


cov = mne.make_ad_hoc_cov(craw.info)

src = mne.read_source_spaces(os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
bem = mne.read_bem_solution(os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'))

mne.viz.plot_alignment(
    craw.info,
    src=src,
    eeg=["original", "projected"],
    trans='fsaverage',
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

fwd = mne.make_forward_solution(craw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=w2o.utils.get_njobs())

mfwd = mne.convert_forward_solution(fwd, force_fixed = False, surf_ori = True, use_cps = True, copy = True)

inv = mne.minimum_norm.make_inverse_operator(info = craw.info, forward = mfwd, noise_cov = cov, loose = 'auto', rank='info', fixed = False, use_cps = True)   # Loose orientations for surface, free for volumes

p_stc = {}
p_stc[ip] = mne.minimum_norm.apply_inverse_epochs(p_epochs[ip], inv, method=method, pick_ori=None, lambda2=lambda2)

labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1')
labels.pop(0)
labels.pop(0)

# For each epoche extract time course and compute psd. Accumulate PSD. Do this for each label and for each period. This will be the cortical spectra. Then as usual.
# Foreach period
# Foreach epoch
# Foreach label
l_tc = p_stc[ip][0].extract_label_time_course(labels, src, mode='mean_flip')
# Compute epoch PSD
# Average
# Eventually normalize on FixRest
# Put everything in "sources" module

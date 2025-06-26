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


all_p_spsd = []
all_p_lbpsd = []

for subject in subjects:
    
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

    # mne.viz.plot_alignment(
    #     craw.info,
    #     src=src,
    #     eeg=["original", "projected"],
    #     trans='fsaverage',
    #     show_axes=True,
    #     mri_fiducials=True,
    #     dig="fiducials",
    #     surfaces=dict(brain=0.4, head=0.1)
    # )

    fwd = mne.make_forward_solution(craw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=w2o.utils.get_njobs())
    
    mfwd = mne.convert_forward_solution(fwd, force_fixed = False, surf_ori = True, use_cps = True, copy = True)
    
    inv = mne.minimum_norm.make_inverse_operator(info = craw.info, forward = mfwd, noise_cov = cov, loose = 1, rank='info', fixed = False, use_cps = True)   # Loose orientations for surface, free for volumes
    
    #labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1')
    #labels.pop(0)
    #labels.pop(0)
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub')
    
    p_spsd = {}
    p_lbpsd = {}
    for ip in p_raws.keys():
        p_lbpsd[ip] = []
        for lb in labels:
            lstc = mne.minimum_norm.compute_source_psd_epochs(    p_epochs[ip], label=lb,
                                                                  inverse_operator=inv, method=method, lambda2=lambda2,
                                                                  fmin=w2o.spectral.get_spectral_limits()[0], fmax=w2o.spectral.get_spectral_limits()[1], bandwidth=2.0, adaptive=False, low_bias=False,
                                                                  pick_ori=None, pca=False, use_cps=True,
                                                                  n_jobs=w2o.utils.get_njobs()
                                                       )
            freqs = lstc[0].times
            p_lbpsd[ip].append(np.mean(np.asarray([np.mean(ls.data, axis=0) for ls in lstc]), axis=0))
    
    for ip in p_raws.keys():    
        
        if norm_period != []:
            p_lbpsd[ip] = [p_lbpsd[ip][l] / p_lbpsd[norm_period][l] for l,lb in enumerate(labels)]
        
        p_spsd[ip] = mne.labels_to_stc(labels, p_lbpsd[ip], tmin=w2o.spectral.get_spectral_limits()[0], tstep=np.unique(np.diff(freqs)), subject='fsaverage', src=src)    
    
    all_p_lbpsd.append(p_lbpsd)
    all_p_spsd.append(p_spsd)

# Prova usando FixRest come covariance anche (con e senza normalizzazione)
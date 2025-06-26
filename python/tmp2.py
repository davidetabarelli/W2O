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

s=0
subject = subjects[s]

craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)

# Get volumes of interest
# CONTROLLALE !!!
def get_vois():
    
    # return [ 'Left-Thalamus-Proper', 'Right-Thalamus-Proper', 
    #          'Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex',
    #          'Left-Caudate', 'Right-Caudate',
    #          'Left-Putamen', 'Right-Putamen',
    #          'Left-Pallidum', 'Right-Pallidum',
    #          'Brain-Stem'
    #         ]
    return [ 'Left-Thalamus', 'Right-Thalamus',              
             'Left-Caudate', 'Right-Caudate',
             'Left-Putamen', 'Right-Putamen',
             'Left-Pallidum', 'Right-Pallidum',
             'Left-Hippocampus', 'Right-Hippocampus',
             'Left-Amygdala', 'Right-Amygdala',
            ]



bem = mne.read_bem_solution(os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'))

ssrc = mne.read_source_spaces(os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
vsrc = mne.setup_volume_source_space('fsaverage',
                                     mri=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'mri', 'aseg.mgz'),
                                     pos=3.0,
                                     volume_label=get_vois(),
                                     add_interpolator=True,
                                     bem=bem,
                                     n_jobs=w2o.utils.get_njobs()
                                     )

src = ssrc + vsrc

cov = mne.make_ad_hoc_cov(craw.info)
fwd = mne.make_forward_solution(craw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=w2o.utils.get_njobs()) 
mfwd = mne.convert_forward_solution(fwd, force_fixed = False, surf_ori = True, use_cps = True, copy = True)
inv = mne.minimum_norm.make_inverse_operator(info = craw.info, forward = mfwd, noise_cov = cov, loose = 1, rank='info', fixed = False, use_cps = True)


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
labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub')

for subject in subjects[::3]:
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=1)
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=1, cov_period='FixRest')

for subject in subjects[1::3]:
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=2)
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=2, cov_period='FixRest')
    
for subject in subjects[2::3]:
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=3)
    w2o.sources.get_periods_source_psds(subject, iperiods, norm_period, method='dSPM', irbio_num=3, cov_period='FixRest')

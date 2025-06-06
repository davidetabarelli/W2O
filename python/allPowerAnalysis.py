import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


s = 4
craw, events, evt_dict = w2o.preliminary.get_clean_data(subjects[s], True)
dbraw = w2o.utils.double_banana_ref(craw)

p_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 1.0)

psds = {}

psds['FixRest'] = p_raws['FixRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['EcRest'] = p_raws['EcRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Masturbation'] = p_raws['Masturbation'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Pleateau'] = p_raws['Pleateau'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Orgasm'] = p_raws['Orgasm'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Resolution'] = p_raws['Resolution'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Breathe2'] = p_raws['Breathe2'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)

freqs = psds['FixRest'].freqs

db_p_raws = {key : w2o.utils.double_banana_ref(p_raws[key].copy()) for key in p_raws.keys()}

db_psds = {}

db_psds['FixRest'] = db_p_raws['FixRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['EcRest'] = db_p_raws['EcRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['Masturbation'] = db_p_raws['Masturbation'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['Pleateau'] = db_p_raws['Pleateau'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['Orgasm'] = db_p_raws['Orgasm'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['Resolution'] = db_p_raws['Resolution'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
db_psds['Breathe2'] = db_p_raws['Breathe2'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)

mne.viz.plot_alignment(
    craw.info,
    src=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'),
    eeg=["original", "projected"],
    trans='fsaverage',
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)
craw.plot_sensors(show_names=True)

fwd = mne.make_forward_solution(
    craw.info, 
    trans='fsaverage', 
    #src=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-vol-5-src.fif'), 
    src=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'), 
    bem=os.path.join(w2o.filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'), 
    eeg=True, 
    mindist=5.0, 
    n_jobs=16
)

noise_cov = mne.compute_raw_covariance(p_raws['FixRest'])

inverse_operator = mne.minimum_norm.make_inverse_operator(p_raws['FixRest'].info, fwd, noise_cov)

stc = mne.minimum_norm.apply_inverse_raw(p_raws['EcRest'], inverse_operator, lambda2=1.0 / 9.0, method='MNE')

hstc = stc.crop(tmin=0, tmax=30).filter(l_freq=8, h_freq=13).apply_hilbert(envelope=True, n_jobs=16)


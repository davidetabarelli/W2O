import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


s = 9
craw, events, evt_dict = w2o.preliminary.get_clean_data(subjects[s], True)
dbraw = w2o.utils.double_banana_ref(craw)

# Periods
p_raws = w2o.preliminary.extract_periods(craw, events, evt_dict, 1.0)
db_p_raws = w2o.preliminary.extract_periods(dbraw, events, evt_dict, 1.0)



# PSDs
periods = w2o.dataset.get_periods_definition().keys()

p_psds = {}

# p_psds['FixOn'] = mne.make_fixed_length_epochs(p_raws['FixRest'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.25).drop_bad().compute_psd(method='welch', fmin=1.0, fmax=45, n_fft=1000, proj=True, n_jobs=16).average()
# p_psds['EcRest'] = mne.make_fixed_length_epochs(p_raws['EcRest'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.25).drop_bad().compute_psd(method='welch', fmin=1.0, fmax=45, n_fft=1000, proj=True, n_jobs=16).average()
# p_psds['Masturbation'] = mne.make_fixed_length_epochs(p_raws['Masturbation'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.25).drop_bad().compute_psd(method='welch', fmin=1.0, fmax=45, n_fft=1000, proj=True, n_jobs=16).average()
# p_psds['Pleateau'] = mne.make_fixed_length_epochs(p_raws['Pleateau'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.25).drop_bad().compute_psd(method='welch', fmin=1.0, fmax=45, n_fft=1000, proj=True, n_jobs=16).average()
# p_psds['Orgasm'] = mne.make_fixed_length_epochs(p_raws['Orgasm'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.25).drop_bad().compute_psd(method='welch', fmin=1.0, fmax=45, n_fft=1000, proj=True, n_jobs=16).average()

p_psds['FixOn'] = mne.make_fixed_length_epochs(p_raws['FixRest'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
p_psds['EcRest'] = mne.make_fixed_length_epochs(p_raws['EcRest'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=False, low_bias=False).pick('eeg').average()
p_psds['Masturbation'] = mne.make_fixed_length_epochs(p_raws['Masturbation'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=True, low_bias=True).pick('eeg').average()
p_psds['Pleateau'] = mne.make_fixed_length_epochs(p_raws['Pleateau'], duration=2, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=True, low_bias=True).pick('eeg').average()
p_psds['Orgasm'] = mne.make_fixed_length_epochs(p_raws['Orgasm'], duration=2.0, proj=True, reject_by_annotation=True, overlap=0.5).drop_bad().compute_psd(method='multitaper', fmin=0.5, fmax=98, bandwidth=2, proj=True, n_jobs=16, adaptive=True, low_bias=True).pick('eeg').average()

freqs = p_psds['FixOn'].freqs

for period in p_psds.keys():
    plt.semilogy(freqs, np.mean(p_psds[period].get_data(), axis=0))
plt.legend(p_psds.keys())




# Vibs
vib_on_epochs = mne.Epochs(craw, mne.merge_events(events, [evt_dict['Vib_test_on'], evt_dict['Vib_1_on'], evt_dict['Vib_2_on']], 200), 200, tmin=0.0, tmax=1.0, baseline=(None,1.0), picks='eeg', reject_by_annotation=True, proj=True)
vib_off_epochs = mne.Epochs(craw, mne.merge_events(events, [evt_dict['Vib_test_off'], evt_dict['Vib_1_off'], evt_dict['Vib_2_off']], 201), 201, 0, 1.0, baseline=(None,1.0), picks='eeg', reject_by_annotation=True, proj=True)

vib_on_epochs.drop_bad()
vib_off_epochs.drop_bad()

vib_on_psd = vib_on_epochs.compute_psd(method='welch', fmin=1.0, fmax=95, n_fft=500, proj=True).average()
vib_off_psd = vib_off_epochs.compute_psd(method='welch', fmin=1.0, fmax=95, n_fft=500, proj=True).average()

freqs = vib_on_psd.freqs

plt.loglog(freqs, np.mean(vib_on_psd.get_data(), axis=0), 'r')
plt.loglog(freqs, np.mean(vib_off_psd.get_data(), axis=0), 'b')
plt.legend(['Vibration on', 'Vibration off'])










psds = {}

psds['FixRest'] = p_raws['FixRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['EcRest'] = p_raws['EcRest'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Masturbation'] = p_raws['Masturbation'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Pleateau'] = p_raws['Pleateau'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Orgasm'] = p_raws['Orgasm'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Resolution'] = p_raws['Resolution'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)
psds['Breathe2'] = p_raws['Breathe2'].compute_psd(fmin=1, fmax=45, n_fft=1024, n_jobs=16, reject_by_annotation=True)

freqs = psds['FixRest'].freqs



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


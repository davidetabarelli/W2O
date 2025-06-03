import numpy as np
import scipy as sp
import os
import PyEMD
import matplotlib.pyplot as plt
import pickle
import mne

import w2o

subjects, N = w2o.dataset.get_subjects()

s = 3
subject = subjects[s]

raw, events, evt_dict = w2o.preliminary.load_raw_data(subject)

bchs_file = os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-badchannels.txt' % subject)
bad_annot_file = os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    

if os.path.exists(bad_annot_file):
    bad_annot = mne.annotations.read_annotations(bad_annot_file)         # CONTROLLA
else:
    bad_annot = mne.annotations.Annotations(0,0.1, description='BAD_MANUAL')   # Insert fake annotation of 0.1 second after start (for display in raw data inspection)
    
if os.path.exists(bchs_file):
    bads = np.loadtxt(fname=bchs_file, dtype='bytes').astype(str).tolist();
    
    # Inject bad channels
    if type(bads) is list:
        raw.info['bads'] = bads
    else:
        raw.info['bads'] = [bads]        
    
    
# Inject manual annotations
raw.set_annotations(bad_annot);

raw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=40e-6))

t_A = raw.annotations[8]['onset'] + raw.annotations[8]['duration']
t_B = raw.annotations[9]['onset']

t_A = 1250
t_B = 1272

#eraw = raw.copy().crop(tmin=t_A, tmax=t_B).pick('eeg').filter(l_freq=1.0, h_freq=None)
eraw = raw.copy().crop(tmin=t_A, tmax=t_B).pick('eeg')
C = len(eraw.info['ch_names'])

eraw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=40e-6))

T = eraw.times
S = eraw.get_data() * 1e6

all_eemds = [];
all_imfs = [];
all_res = [];
for c in range(C):
    
    eemd = PyEMD.EEMD(parallel=True)
    eemd.eemd(S[c,:], T, progress=True)
    
    imfs, res = eemd.get_imfs_and_residue()
    
    all_imfs.append(imfs)
    all_eemds.append(eemd)
    all_res.append(res)
    
    del eemd



# Save
res = {'all_imfs': all_imfs, 'all_eemds': all_eemds, 'all_res': all_res, 't_A': t_A, 't_B': t_B, 'subject': subject, 's': s}
with open('eemd_res.pkl', 'wb') as f:
    pickle.dump(res, f)
    

# Inspect and mark
art_imfs_map = []
for c in range(C):
    
    ch_name = eraw.info['ch_names'][c]
    
    M = all_imfs[c].shape[0]
    
    #i_info = eraw.copy().drop_channels(eraw.info['ch_names'][M+1:]).info
    #i_info.rename_channels({i_info['ch_names'][j] : 'IMF_%s_%02d' % (ch_name, j) for j in range(M+1)})
    #i_info.rename_channels({'IMF_%s_%02d' % (ch_name, M) : 'RES_%s' % ch_name})
    #iraw = mne.io.RawArray(np.vstack((all_imfs[c], all_res[c]))*1e-6, i_info)
    
    i_info = eraw.copy().drop_channels(eraw.info['ch_names'][M:]).info
    i_info.rename_channels({i_info['ch_names'][j] : 'IMF_%s_%02d' % (ch_name, j) for j in range(M)})
    iraw = mne.io.RawArray(all_imfs[c]*1e-6, i_info)
    ipsd = iraw.compute_psd(fmin=7, fmax=75, n_jobs=8)
    
    #ipsd.plot_topo()
    #iraw.plot(block=True)
    
    art_imfs_map.append(np.asarray(np.argwhere(np.asarray([sp.stats.linregress(ipsd.freqs, ipsd.get_data()[i,:], 'greater').pvalue for i in range(M)]) <= 0.05)).reshape(-1,))
    #iraw.info['bads'] = [iraw.info['ch_names'][bc] for bc in art_imfs_map[c]]
    #iraw.plot(block=True)
    

Y = 1e-6*np.asarray([all_imfs[c][i] for i in art_imfs_map[c] for c in range(C)])
R = Y.shape[0]

craw = mne.io.RawArray(Y, mne.create_info(['Fc_%3d' % j for j in range(R)], 500.0, 'eeg'))
craw.set_montage(mne.channels.make_dig_montage({craw.info['ch_names'][r] : [1e-1*(R/2-r)/R, 1e-1*(R/2-r)/R, 0] for r in range(R)}))

ica = mne.preprocessing.ICA(n_components=128, random_state=19579, method='picard', max_iter=10000)
ica.fit(craw)

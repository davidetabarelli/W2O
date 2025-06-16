# All functions for preliminary analysis

import mne;
import os;
import numpy as np;
import pickle;

from w2o import filesystem
from w2o import dataset
from w2o import utils



########## FUNCTIONS
# Load non preprocessed data with standard filters and montage
def load_raw_data(subject):
    
    # Load raw data
    raw = mne.io.read_raw_brainvision((os.path.join(filesystem.get_eegrawsubjectdir(subject), '%s_eeg.vhdr' % subject)), preload=True)
    
    # Base filters
    raw.filter(l_freq=0.5, h_freq=95)
    raw.notch_filter(freqs=[50, 100], notch_widths=2.0)
    
    # Set channel types
    #
    # TODO: (WARNING) to be checked, in some subjects is not consistent with the labels from matlab script !!!!!
    #
    ch_types = {ch : 'eeg' for ch in raw.info['ch_names']}
    ch_types['HEOG_left'] = 'eog'
    ch_types['HEOG_right'] = 'eog'
    ch_types['VEOG_right'] = 'eog'
    ch_types['LL'] = 'emg'
    ch_types['RA'] = 'emg'
    raw.set_channel_types(ch_types)

    # Set montage
    raw.set_montage('standard_1005')
    
    # Parse and cleanup events/annotations
    _, oevt_id = mne.events_from_annotations(raw)
    evt_dict = {evt: int(evt.split('/')[1]) for evt in oevt_id.keys() if evt.split('/')[0] == 'Stimulus'}
    events, evt_dict = mne.events_from_annotations(raw, event_id=evt_dict)
    events = mne.merge_events(events, [101,103], 101)       # Merge instructions end (button/no button)
    evt_dict = dataset.get_event_dict()
    raw.set_annotations(None)
        
    # Check first sample and first time are set to zero (simpler events and annotations management)
    assert raw.first_samp == 0
    assert raw.first_time == 0
    
    # Correct 75 event issue or resolution
    if not np.any(events[:,2] == 75):
        idx = np.argwhere(events[np.argwhere(events[:,2] == 73).reshape(-1)[0]:,2] == 999).reshape(-1)[0] + np.argwhere(events[:,2] == 73).reshape(-1)[0]
        nevent = events[idx,:]        
        events = np.insert(events, idx, nevent, axis=0)
        events[idx,2] = 75
        
    
    # Check
    #assert np.all(np.unique(np.asarray([v for v in evt_dict.values()])) == np.unique(events[:,2]))
    assert(np.all(np.isin(np.unique(events[:,2]), np.asarray([v for v in evt_dict.values()]))))         # Alcuni eventi possono essere assenti
    
    # Finally crop extremal: WARNING : THIS CHN
    #raw.crop(tmin=(events[0,0]/raw.info['sfreq']) - 1, tmax=(events[-1,0]/raw.info['sfreq']) + 1)
    
    # TODO: quando mi dicono bene che succede (vedi file domande)
    # Create not usable (as BAD_NO_USE) periods as annotations. So far "Instructions"
    #bad_evt = mne.pick_events(events, include=[evt_dict[key] for key in ['Instructions_on', 'Instructions_off']])
        
    return raw, events, evt_dict


def preprocess_data(subject):
    
    # Load data and events
    raw, events, evt_dict = load_raw_data(subject)
        
    # Files
    bchs_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-badchannels.txt' % subject)
    bad_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
    muscle_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-muscle-annotations.fif' % subject)
    muscle_enhc_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-muscle-enhc-annotations.fif' % subject)
    ica_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject)
    ica_excl_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-excluded.txt' % subject)
    ica_enhc_excl_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-enhc-excluded.txt' % subject)
    
    # Load previous manual annotations and badchannels (if present)
    if os.path.exists(bad_annot_file):
        bad_annot = mne.annotations.read_annotations(bad_annot_file)
    else:
        bad_annot = mne.annotations.Annotations(0.1,0.5, description='MANUAL')
    
    if os.path.exists(muscle_annot_file):
        muscle_annot = mne.annotations.read_annotations(muscle_annot_file)
    else:        
        muscle_annot = mne.annotations.Annotations(0.1,0.5, description='MUSCLE')
        
    if os.path.exists(bchs_file):
        bads = np.loadtxt(fname=bchs_file, dtype='bytes').astype(str).tolist();
        
        # Inject bad channels
        if type(bads) is list:
            raw.info['bads'] = bads
        else:
            raw.info['bads'] = [bads]        
        
        
    # Inject manual annotations
    raw.set_annotations(bad_annot + muscle_annot);
    
    # Mark bad channels and bad segments
    raw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=40e-6))
    
    # Save bad annotations
    bad_annot = raw.annotations[np.argwhere([ann['description'] == 'MANUAL' for ann in raw.annotations]).reshape(-1)]
    bad_annot.save(bad_annot_file, overwrite = True)
    muscle_annot = raw.annotations[np.argwhere([ann['description'] == 'MUSCLE' for ann in raw.annotations]).reshape(-1)]
    muscle_annot.save(muscle_annot_file, overwrite = True)
    
    # Save bad channels
    np.savetxt(fname=bchs_file, X=raw.info['bads'], fmt='%s')
    
    # Interpolate bad channels
    raw.load_data().interpolate_bads()
    
    # Compute average ref
    rraw = utils.ref_to_avg_with_fcz(raw, True)
    
    
    # ICA
    iraw = rraw.copy().filter(l_freq=0.5, h_freq=95).pick('eeg')
    iraw.set_annotations(bad_annot.copy().rename({'MANUAL': 'BAD_MANUAL'}))
    if os.path.exists(ica_file):
        ica = mne.preprocessing.read_ica(ica_file)
        ica.exclude = []
    else:
        ica = mne.preprocessing.ICA(n_components=0.999999, random_state=19579, method='fastica', max_iter=10000000, fit_params=dict(max_iter=10000000, tol=1e-12 ))
        ica.fit(iraw, reject_by_annotation=True)
        ica.save(ica_file, overwrite = True)
    
    
    # No muscle inspection
    if os.path.exists(ica_excl_file):
        ica_excluded = np.loadtxt(fname=ica_excl_file, dtype='bytes').astype(int).tolist();
    else:
        ica.plot_components(inst=iraw, nrows=6, ncols=10, res=48)
        ica.plot_sources(iraw, block=True)
        ica_excluded = ica.exclude
        np.savetxt(fname=ica_excl_file, X=ica_excluded, fmt='%s')
    
    # Muscolar inspection
    if os.path.exists(ica_enhc_excl_file):
        ica_enhc_excluded = np.loadtxt(fname=ica_enhc_excl_file, dtype='bytes').astype(int).tolist();
    else:
        ica.plot_components(inst=iraw, nrows=6, ncols=10, res=48)
        ica.plot_sources(iraw, block=True)
        ica_enhc_excluded = np.setxor1d(ica.exclude, ica_excluded)
        np.savetxt(fname=ica_enhc_excl_file, X=ica_enhc_excluded, fmt='%s')
    
    
    # Join back excluded
    ica.exclude = list(np.hstack((ica_excluded, ica_enhc_excluded)))
    
    
    # Last residual muscolar enhanced annotations
    if os.path.exists(muscle_enhc_annot_file):
        muscle_enhc_annot = mne.annotations.read_annotations(muscle_enhc_annot_file)
    else:
        muscle_enhc_annot = mne.annotations.Annotations(0,0.1, description='MUSCLE_ENHC')
    
        craw = ica.apply(rraw)
        craw.set_annotations(bad_annot + muscle_enhc_annot)    
        craw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=40e-6))
        
        muscle_enhc_annot = craw.annotations[np.argwhere([ann['description'] == 'MUSCLE_ENHC' for ann in craw.annotations]).reshape(-1)]
        muscle_enhc_annot.save(muscle_enhc_annot_file, overwrite = True)


    
    


def get_clean_data(subject, enhc_flag=True):
    
    # Load data and events
    raw, events, evt_dict = load_raw_data(subject)
    
    # Files
    bchs_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-badchannels.txt' % subject)
    bad_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
    muscle_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-muscle-annotations.fif' % subject)
    muscle_enhc_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-muscle-enhc-annotations.fif' % subject)
    ica_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject)
    ica_excl_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-excluded.txt' % subject)
    ica_enhc_excl_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-enhc-excluded.txt' % subject)
    
            
    # Load and inject bad channels
    bads = np.loadtxt(fname=bchs_file, dtype='bytes').astype(str).tolist();
    if type(bads) is list:
        raw.info['bads'] = bads
    else:
        raw.info['bads'] = [bads]     
    
    
    # Interpolate bad channels
    raw.load_data().interpolate_bads()
        
    # Compute average ref
    rraw = utils.ref_to_avg_with_fcz(raw, True)
    
    # Load and apply ICA
    ica = mne.preprocessing.read_ica(ica_file)
    ica.exclude = []
    if enhc_flag:
        ica_enhc_excluded = np.loadtxt(fname=ica_enhc_excl_file, dtype='bytes').astype(int).tolist();
        ica_excluded = np.loadtxt(fname=ica_excl_file, dtype='bytes').astype(int).tolist();
        ica.exclude = list(np.hstack((ica_excluded, ica_enhc_excluded)))
    else:        
        ica_excluded = np.loadtxt(fname=ica_excl_file, dtype='bytes').astype(int).tolist();
        ica.exclude = ica_excluded
    
    craw = ica.apply(rraw)
    
    # Load and inject annotations
    bad_annot = mne.annotations.read_annotations(bad_annot_file)
    if enhc_flag:
        muscle_enhc_annot = mne.annotations.read_annotations(muscle_enhc_annot_file)
        craw.set_annotations(bad_annot.copy().rename({'MANUAL': 'BAD_MANUAL'}) + muscle_enhc_annot.copy().rename({'MUSCLE_ENHC': 'BAD_MUSCLE_ENHC'}))
    else:
        muscle_annot = mne.annotations.read_annotations(muscle_annot_file)
        craw.set_annotations(bad_annot.copy().rename({'MANUAL': 'BAD_MANUAL'}) + muscle_annot.copy().rename({'MUSCLE': 'BAD_MUSCLE'}))
    
    
    return craw, events, evt_dict


def extract_periods(craw, events, evt_dict, time_tol=2.0, periods=[]):
    
    Fs = craw.info['sfreq']
    
    periods_def = dataset.get_periods_definition()
    
    if not periods == []:
        periods_def = {ip : periods_def[ip] for ip in periods}
    
    p_raws = {}
    
    for key, value in periods_def.items():
        
        print('Extracting %s' % key)
        
        evt_1 = evt_dict[value['evt_1']]
        evt_2 = evt_dict[value['evt_2']]
        
        t_evts = mne.pick_events(events, include=[evt_1, evt_2])
        
        assert np.all(t_evts[1:,0] >= t_evts[:-1,0])
        
        i1 = np.argwhere(t_evts[:,2] == evt_1).reshape(-1)[0]
        i2 = np.argwhere(t_evts[i1:,2] == evt_2).reshape(-1)[0] + i1
        
        assert t_evts[i1,2] == evt_1
        assert t_evts[i2,2] == evt_2    
        
        if np.isin(key, ['VibTest', 'Muscles', 'Vib1', 'Vib2']):
            t1 = t_evts[i1,0] / Fs - 0.1
            t2 = t_evts[i2,0] / Fs + 0.1
        else:
            t1 = t_evts[i1,0] / Fs + time_tol
            t2 = t_evts[i2,0] / Fs - time_tol
        
        p_raws[key] = craw.copy().crop(tmin=t1, tmax=t2)
    
    return p_raws
  


def extract_muscles_epochs(praw, events, evt_dict, duration=2, overlap=0.5):
    
    Fs =  praw.info['sfreq']
    
    f_samp = praw.first_samp
    l_samp = praw.last_samp
    
    levents = events[np.argwhere((events[:,0] >= f_samp) & (events[:,0] <= l_samp)).reshape(-1),:]
    levents = mne.pick_events(levents, include=[evt_dict['Muscles_start'], evt_dict['Routine_end']])
    
    assert(np.unique(levents[0::2,2])[0] == evt_dict['Muscles_start'])
    assert(np.unique(levents[1::2,2])[0] == evt_dict['Routine_end'])
    
    all_epochs = []
    for i in range(int(levents.shape[0]/2)):
        
        i1 = 2*i;
        i2 = 2*i + 1;
        
        t1 = (levents[i1,0] - f_samp) / Fs
        t2 = (levents[i2,0] - f_samp) / Fs
        
        all_epochs.append(mne.make_fixed_length_epochs(praw.copy().crop(tmin=t1, tmax=t2), duration=duration, proj=True, reject_by_annotation=True, overlap=overlap).drop_bad())
    
    epochs = mne.concatenate_epochs(all_epochs)
    
    return epochs
        
    
    
def extract_vib_epochs(praw, events, evt_dict):
    
    Fs =  praw.info['sfreq']
    
    f_samp = praw.first_samp
    l_samp = praw.last_samp
    
    levents = events[np.argwhere((events[:,0] >= f_samp) & (events[:,0] <= l_samp)).reshape(-1),:]
    levents = mne.pick_events(levents, include=[evt_dict['Vib_test_on'], evt_dict['Vib_test_off'], evt_dict['Vib_1_on'], evt_dict['Vib_1_off'], evt_dict['Vib_2_on'], evt_dict['Vib_2_off']])
    
    if np.any(np.isin(np.unique(levents[:,2]), evt_dict['Vib_test_on'])):
        v_on_evt = evt_dict['Vib_test_on']
        v_off_evt = evt_dict['Vib_test_off']
        
    if np.any(np.isin(np.unique(levents[:,2]), evt_dict['Vib_1_on'])):
        v_on_evt = evt_dict['Vib_1_on']
        v_off_evt = evt_dict['Vib_1_off']
    
    if np.any(np.isin(np.unique(levents[:,2]), evt_dict['Vib_2_on'])):
        v_on_evt = evt_dict['Vib_2_on']
        v_off_evt = evt_dict['Vib_2_off']
    
    assert(np.unique(levents[0::2,2])[0] == v_on_evt)
    assert(np.unique(levents[1::2,2])[0] == v_off_evt)
    
    assert np.min((np.diff(levents[:,0]) / Fs)[0::2]) >= 0.95 # Minimal length with tolerance
    assert np.min((np.diff(levents[:,0]) / Fs)[1::2]) >= 1.45
    
    levents = mne.merge_events(mne.merge_events(levents, [22, 40, 60], 500), [23, 41, 61], 501)
    
    epochs_on =  mne.Epochs(praw, levents, event_id={'VibOn' : 500}, tmin=0.0, tmax=0.998, baseline=(None,None), reject_by_annotation=True, preload=True)
    epochs_off = mne.Epochs(praw, levents, event_id={'VibOff' : 501}, tmin=0.5, tmax=1.498, baseline=(None,None), reject_by_annotation=True, preload=True)
    
    if  (len(epochs_off) == 0) & (len(epochs_on) == 0):
        return
        
    if not len(epochs_off) == 0:
        epochs_off.shift_time(-0.5)
        epochs_off.baseline = epochs_on.baseline
    
    epochs_on.drop_bad()
    epochs_off.drop_bad()
    
    epochs = mne.concatenate_epochs([epochs_on, epochs_off])
    
    return epochs

    
    


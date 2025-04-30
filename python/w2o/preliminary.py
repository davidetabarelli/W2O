# All functions for preliminary analysis

import mne;
import os;
import numpy as np;

from w2o import filesystem
from w2o import dataset



########## FUNCTIONS
# Load non preprocessed data with standard filters and montage
def load_raw_data(subject):
    
    # Load raw data
    raw = mne.io.read_raw_brainvision((os.path.join(filesystem.get_eegrawsubjectdir(subject), '%s_eeg.vhdr' % subject)), preload=True)
    
    # Base filters
    raw.filter(l_freq=0.2, h_freq=225)
    raw.notch_filter(freqs=[50, 100, 150, 200], notch_widths=2.0)
    
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
    
    # Check
    assert np.all(np.unique(np.asarray([v for v in evt_dict.values()])) == np.unique(events[:,2]))
    
    # Check first sample and first time are set to zero (simpler events and annotations management)
    assert raw.first_samp == 0
    assert raw.first_time == 0
    
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
    bchs_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-.badchannels.txt' % subject)
    bad_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
    ica_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject)
    
    # Load previous manual annotations (if present)
    if os.path.exists(bad_annot_file):
        bad_annot = mne.annotations.read_annotations(bad_annot_file)         # CONTROLLA
    else:
        bad_annot = mne.annotations.Annotations(0,0.1, description='BAD_MANUAL')   # Insert fake annotation of 0.1 second after start (for display in raw data inspection)
        
    # Inject manual annotations
    raw.set_annotations(bad_annot);
    
    # Mark bad channels and bad segments
    raw.plot(event_color='r', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=40e-6))
    
    # Save bad annotations
    bad_annot = raw.annotations;
    bad_annot.save(bad_annot_file, overwrite = True)
    
    # Save bad channels
    np.savetxt(fname=bchs_file, X=raw.info['bads'], fmt='%s')
    
    # Interpolate bad channels
    raw.load_data().interpolate_bads()
    
    # Data for ICA
    iraw = raw.copy().filter(l_freq=0.2, h_freq=None).pick('eeg')
    
    # Fit or load ICA if existing    
    if os.path.exists(ica_file):
        ica = mne.preprocessing.read_ica(ica_file)
    else:
        # Create ICA object
        #ica = mne.preprocessing.ICA(n_components=0.99, random_state=19579, method='infomax', fit_params=dict(extended=True))
        ica = mne.preprocessing.ICA(n_components=48, random_state=19579, method='fastica')
        
        # Fit ICA to data        
        ica.fit(iraw, reject_by_annotation=True)
    
        # Save raw ICA
        ica.save(ica_file, overwrite = True)
        
    # Score automatically where possible (suggest)
    if len(ica.exclude) == 0:
        eog_idx, eog_scores = ica.find_bads_eog(raw)
        ecg_idx, ecg_scores = ica.find_bads_ecg(raw)
        ica.exclude = np.unique(eog_idx + ecg_idx)
    
    
    # Display ICA for inspection
    ica.plot_components(inst=iraw, nrows=6, ncols=8, res=48)
    ica.plot_sources(iraw, block=True)
    
    # Save inspection
    ica.save(ica_file, overwrite = True)
    
            
    return

def get_clean_data(subject):
    
    # Load data and events
    raw, events, evt_dict = load_raw_data(subject)

    # Files
    bchs_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-.badchannels.txt' % subject)
    bad_annot_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
    ica_file = os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject)

    # Load manual annotations and bad channels
    bad_annot = mne.annotations.read_annotations(bad_annot_file) 
    bads = np.loadtxt(fname=bchs_file, dtype='bytes').astype(str).tolist();
    
    # Inject bad channels
    if type(bads) is list:
        raw.info['bads'] = bads
    else:
        raw.info['bads'] = [bads]
    
    # Inject annotations
    raw.set_annotations(bad_annot)
    
    # Interpolate bad channels
    raw.load_data().interpolate_bads()
    
    # Load ICA
    ica = mne.preprocessing.read_ica(ica_file)
    
    # Apply ICA
    eeg_raw = raw.copy().pick('eeg')
    eog_raw = raw.copy().pick('eog')
    ecg_raw = raw.copy().pick(['ecg'])    
    
    # Apply ICA removing bad components on each sensor type
    craw = ica.apply(eeg_raw)    
    
    # Join back data
    craw.add_channels([eog_raw, ecg_raw])
    
    return craw, events, evt_dict

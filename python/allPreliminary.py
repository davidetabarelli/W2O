import numpy as np
import scipy as sp
import os

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()

 
# Manual preprocessing
for subject in subjects:
    w2o.preliminary.preprocess_data(subject)



# for subject in subjects:
    
#     bad_annot_file = os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
#     craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
#     craw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=25e-6))
#     bad_annot = craw.annotations[np.argwhere([ann['description'] == 'BAD_MANUAL' for ann in craw.annotations]).reshape(-1)]
#     bad_annot.rename({'BAD_MANUAL': 'MANUAL'})    
#     bad_annot.save(bad_annot_file, overwrite = True)

    
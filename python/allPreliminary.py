import numpy as np
import scipy as sp
import os

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()


# Codice ispezioni preliminari
#for subject in subjects:
#    raw,_,_ = w2o.preliminary.load_raw_data(subject); 
#    raw.pick(['eog', 'emg'])
#    raw.plot(block=True)
# 0 VEOG-RA
# 4 Not usable LL

# Manual preprocessing
#for subject in subjects:
#    w2o.preliminary.preprocess_data(subject)
    




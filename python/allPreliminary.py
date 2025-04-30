import numpy as np
import scipy as sp
import os

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()

evt_dict = w2o.dataset.get_event_dict()


s = 0
subject = subjects[s]

w2o.preliminary.preprocess_data_manual(subject)



####
raw.plot(event_color='r', events=events, event_id=evt_dict)


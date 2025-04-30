import numpy as np
import scipy as sp
import os

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()




s = 2
subject = subjects[s]

w2o.preliminary.preprocess_data(subject)



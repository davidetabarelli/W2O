from w2o import filesystem
import mne

########## FUNCTIONS ##########
def initialize():
    print('')
    mne.set_config('SUBJECTS_DIR', filesystem.get_anatomydir())


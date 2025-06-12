import numpy as np
import scipy as sp
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import mne

import w2o


subjects, N = w2o.dataset.get_subjects()

 
# FATTO
# Manual preprocessing
# for subject in subjects:
#    w2o.preliminary.preprocess_data(subject)

# FATTO
# for subject in subjects:
    
#     bad_annot_file = os.path.join(w2o.filesystem.get_artfctsubjectdir(subject), '%s-bad-annotations.fif' % subject)    
#     craw, events, evt_dict = w2o.preliminary.get_clean_data(subject, True)
#     craw.plot(event_color='b', events=events, event_id=evt_dict, block=True, remove_dc=True, n_channels=59, duration=20, scalings=dict(eeg=25e-6))
#     bad_annot = craw.annotations[np.argwhere([ann['description'] == 'BAD_MANUAL' for ann in craw.annotations]).reshape(-1)]
#     bad_annot.rename({'BAD_MANUAL': 'MANUAL'})    
#     bad_annot.save(bad_annot_file, overwrite = True)


# Final dataset info
dinfo = w2o.dataset.get_dataset_info()


# Figura ditribuzione tempi dei periodi rilevanti
periods = [period for period in w2o.dataset.get_periods_definition().keys()]
iperiods = ['FixRest', 'EcRest', 'Breathe1', 'Masturbation', 'Pleateau', 'Orgasm', 'Resolution', 'Breathe2']
df = dinfo.melt(id_vars='Subject', value_vars=[col for col in dinfo.columns if col.endswith('_Time')], var_name='Period', value_name='Time').assign(Period=lambda d: d['Period'].str.replace('_Time', '', regex=False))

sns.set_theme(style="white", rc={"axes.spines.right": False, "axes.spines.top": False})
fig, ax = plt.subplots()

sns.stripplot(data=df.query("Period in @iperiods"), x="Time", y="Period", dodge=False, alpha=.35, legend=False, size=6)

ax.set_xlabel('Time [s]')
ax.set_ylabel('')
fig.set_size_inches([10.5,5])


for ip in iperiods:
    
    pmin = df.query('Period==@ip')['Time'].min()
    pmax = df.query('Period==@ip')['Time'].max()
    
    y_pos = iperiods.index(ip)
    
    plt.text(pmin-15, y_pos+0.1, f"{pmin:.1f}", ha='center', va='bottom', fontsize=8)
    plt.text(pmax+15, y_pos+0.1, f"{pmax:.1f}", ha='center', va='bottom', fontsize=8)

fig.suptitle('Distribuzione lunghezze periodi di interesse. Ogni punto è un soggetto. I valori indicano minimo e massimo sul campione per quel periodo.', fontsize=10)

ax.set_xlim((ax.get_xlim()[0] - 10,ax.get_xlim()[1] + 10))

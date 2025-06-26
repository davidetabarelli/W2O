# Utilities
import numpy as np
import scipy as sp
import os
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
import fabric
import paramiko
import joblib
import datetime

import mne
from w2o import preliminary
from w2o import filesystem
from w2o import dataset
from w2o import utils
from w2o import spectral



#### FUNCTIONS

# Prepare remote script based on inputs
def prepare_remote_psd_script(unique_str, method, lambda2, fmin, fmax, n_jobs=140):    
    
    # sc_str =  ""
    # sc_str += "\n"
    # sc_str += "import numpy as np\n"
    # sc_str += "import mne\n"
    # sc_str += "import joblib\n"
    # sc_str += "\n"
    # sc_str += "data_in = joblib.load('w2o_remote_tmp_data_in_%s.joblib')\n" % unique_str
    # sc_str += "\n"
    # sc_str += "p_lb_spds = {}\n"
    # sc_str += "freqs = {}\n"
    # sc_str += "for ip in data_in['p_epochs'].keys():\n"
    # sc_str += "\tp_lb_spds[ip] = []\n"    
    # sc_str += "\tlstc = mne.minimum_norm.compute_source_psd_epochs(data_in['p_epochs'][ip], inverse_operator=data_in['inv'], method=\"%s\", lambda2=%f, fmin=%f, fmax=%f, bandwidth=2.0, adaptive=False, low_bias=False, pick_ori=None, pca=False, use_cps=True, n_jobs=%d)\n" % (method, lambda2, fmin, fmax, n_jobs)
    # sc_str += "\tfreqs[ip] = lstc[0].times\n"
    # #sc_str += "\tp_lb_psds[ip].append(np.asarray([mne.extract_label_time_course(ls, data_in['labels'], data_in['src'], mode='mean_flip') for ls in lstc]))\n"    
    # sc_str += "\tp_lb_spds[ip] = np.asarray([mne.extract_label_time_course(ls, data_in['labels'], data_in['src'], mode='mean') for ls in lstc])\n"    
    # sc_str += "\n"
    # sc_str += "data_out = {'p_lb_spds': p_lb_spds, 'freqs': freqs}\n"
    # sc_str += "joblib.dump(data_out, './w2o_remote_tmp_data_out_%s.joblib', compress=3)\n" % unique_str
    # sc_str += "\n"
    # sc_str += "quit()\n"
    
    sc_str =  ""
    sc_str += "\n"
    sc_str += "import numpy as np\n"
    sc_str += "import mne\n"
    sc_str += "import joblib\n"
    sc_str += "\n"
    sc_str += "data_to = joblib.load('w2o_remote_tmp_data_to_%s.joblib')\n" % unique_str
    sc_str += "\n"
    sc_str += "lstc = mne.minimum_norm.compute_source_psd_epochs(data_to['epochs'], inverse_operator=data_to['inv'], method=\"%s\", lambda2=%f, fmin=%f, fmax=%f, bandwidth=2.0, adaptive=False, low_bias=False, pick_ori=None, pca=False, use_cps=True, n_jobs=%d)\n" % (method, lambda2, fmin, fmax, n_jobs)
    sc_str += "freqs = lstc[0].times\n"    
    sc_str += "lb_psd = np.asarray([mne.extract_label_time_course(ls, data_to['labels'], data_to['src'], mode='mean') for ls in lstc])\n"    
    sc_str += "\n"
    sc_str += "data_from = {'lb_psd': lb_psd, 'freqs': freqs}\n"
    sc_str += "joblib.dump(data_from, './w2o_remote_tmp_data_from_%s.joblib', compress=3)\n" % unique_str
    sc_str += "\n"
    sc_str += "quit()\n"
    
    return sc_str


# Compute, on selected IRBIO remotely, the spectrum of selected epochs. Return non averaged and non normalized. If already saved load from disk
def compute_period_source_psd(subject, period, epochs, inv, src, labels, fmin=None, fmax=None, method='eLORETA', irbio_num=3, n_jobs=140, cov_period=None):
   
    # Deduci atlas
    if len(labels) >= 400:
        atlas = 'aparc_sub'
    else:
        atlas = 'HCPMMP'
    
    # Results file
    if cov_period == None:
        out_file = os.path.join(filesystem.get_resultssubjectdir('spectral', subject), "source_spectra_%s_%s_%s_%.1f_to_%.1f_%s.joblib" % (atlas, subject, period, fmin, fmax, method))
    else:
        out_file = os.path.join(filesystem.get_resultssubjectdir('spectral', subject), "source_spectra_%s_%s_%s_%.1f_to_%.1f_%s_cov_%s.joblib" % (atlas, subject, period, fmin, fmax, method, cov_period))
    
    # If exists load data
    if os.path.exists(out_file):
        
        # Load and parse data
        res = joblib.load(out_file)        
        lb_psd = res['lb_psd']
        freqs = res['freqs']
        
    else:
            
        # Source inversion parameters
        snr = 1  # Evoked (average) activity intversion
        lambda2 = 1/snr**2
        
        # Unique string for files
        now = datetime.datetime.now(); 
        unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
        
        # Remote files names
        script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
        data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
        data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
        
        # Frequency limits
        if fmin == None:
            fmin = spectral.get_spectral_limits()[0]
        if fmax == None:
            fmax = spectral.get_spectral_limits()[1]
            
        # Open remote connection
        conn = filesystem.open_remote_connection(irbio_num)
        
        # Prepare data
        data_to = {'epochs': epochs, 'labels': labels, 'inv': inv, 'src': src}
        
        # Save data
        joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
        
        # Prepare and save script
        with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
            f.write(prepare_remote_psd_script(unique_str, method, lambda2, fmin, fmax, 140))
            
        # Transfer data and script
        conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
        conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
        
        # Run remote command
        conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
        
        # Transfer back results
        conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
        
        # Read results
        data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
        lb_psd = data_from['lb_psd']
        freqs = data_from['freqs']
            
        # Delete files and variables
        del data_to
        del data_from
        os.remove(os.path.join(filesystem.get_local_wd(), script_file))
        os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
        os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
        # Close connection
        filesystem.close_remote_connection(conn)
        
        # Save data
        res = {'lb_psd': lb_psd, 'freqs': freqs, 'labels': labels, 'method': method, 'lambda2': lambda2}
        joblib.dump(res, out_file, compress=3)
        
    
    return lb_psd, freqs
 


# Get periods spectral PSDS (on aparc_sub atlas)
def get_periods_source_psds(subject, periods=[], norm_period=[], method='eLORETA', irbio_num=3, cov_period=None):
       
    # Get periods of interest
    if periods == []:
        iperiods = [k for k in dataset.get_periods_definition().keys()]
    else:
        iperiods = periods
        
    # Frequency bands
    f_bands = spectral.get_fbands_dict()
    
    # Check if all periods on selected options are already computed
    # TODO: spare time
    
    
    # Get craw
    craw, events, evt_dict = preliminary.get_clean_data(subject, True)
    
    # Pick EEG channels
    craw.pick('eeg')
    
    # Get periods
    p_raws = preliminary.extract_periods(craw, events, evt_dict, 0, iperiods)

    # Create epochs
    p_epochs = {}
    for ip in p_raws.keys():
        
        if ip == 'Muscles':
            
            lepochs = preliminary.extract_muscles_epochs(p_raws[ip], events, evt_dict, 2.0, 0.5)
                    
        elif np.isin(ip, ['VibTest', 'Vib1', 'Vib2']):
            
            lepochs = preliminary.extract_vib_epochs(p_raws[ip], events, evt_dict)
            
        else:
            lepochs = mne.make_fixed_length_epochs(p_raws[ip], duration=2, proj=True, reject_by_annotation=True, overlap=0.5)
            lepochs.drop_bad()
        
        p_epochs[ip] = lepochs
    
    
    # Join Vibs periods adding a new one
    if np.all(np.isin(['VibTest', 'Vib1', 'Vib2'], iperiods)):
        p_epochs['VibOn'] = mne.concatenate_epochs([p_epochs[ip]['VibOn'] for ip in ['VibTest', 'Vib1', 'Vib2'] if p_epochs[ip] != None])
        p_epochs['VibOff'] = mne.concatenate_epochs([p_epochs[ip]['VibOff'] for ip in ['VibTest', 'Vib1', 'Vib2'] if p_epochs[ip] != None])
    
        # Prune old Vib epochs
        [p_epochs.pop(vp) for vp in ['VibTest', 'Vib1', 'Vib2']]
    
    # Prepare inversion kernel
    if cov_period == None:        
        cov = mne.make_ad_hoc_cov(craw.info)
    else:
        # Compute covariance on epochs (all or FixRest?) and keep only diagonal elements. See: https://neuroimage.usc.edu/brainstorm/Tutorials/NoiseCovariance#Variations_on_how_to_estimate_sample_noise_covariance
        cov = mne.compute_covariance(p_epochs[cov_period])
        cov.as_diag()
    
    src = mne.read_source_spaces(os.path.join(filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    bem = mne.read_bem_solution(os.path.join(filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif'))
    fwd = mne.make_forward_solution(craw.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=utils.get_njobs()) 
    mfwd = mne.convert_forward_solution(fwd, force_fixed = False, surf_ori = True, use_cps = True, copy = True)
    inv = mne.minimum_norm.make_inverse_operator(info = craw.info, forward = mfwd, noise_cov = cov, loose = 1, rank='info', fixed = False, use_cps = True)   # Loose orientations for surface, free for volumes
    labels = mne.read_labels_from_annot('fsaverage', parc='aparc_sub')[:448]
    
    # Cycle trough periods and remote-compute raw psds on labels
    p_lb_spds = {ip: None for ip in iperiods}
    freqs = {ip: None for ip in iperiods}
    for ip in iperiods:
        lb_psd, lfreqs = compute_period_source_psd(subject, ip, p_epochs[ip], inv, src, labels, fmin=spectral.get_spectral_limits()[0], fmax=spectral.get_spectral_limits()[1], method=method, irbio_num=irbio_num, n_jobs=140, cov_period=cov_period)
        p_lb_spds[ip] = lb_psd
        freqs[ip] = lfreqs
    
    
    # If selected normalize PSD on selected normalization period 
    if norm_period != []:
        bsln = np.mean(p_lb_spds[norm_period], axis=0)
        for ip in p_lb_spds.keys():
            ff_idx = np.argwhere(np.isin(freqs[norm_period], freqs[ip])).reshape(-1)
            ndata = (p_lb_spds[ip] / bsln[np.newaxis,:,ff_idx])
            p_lb_spds[ip] = ndata
    
    # Average
    avg_p_lb_spds = {ip : np.mean(p_lb_spds[ip], axis=0) for ip in p_lb_spds.keys()}
    
    # TODO: frequency bands ..
    
    return p_lb_spds, avg_p_lb_spds, freqs, labels
    
    # If labels is a full atlas set the name for disk load (faster)
def get_labels_adjacency(src, labels, atlas=None):
    
    computeAdj = True
    
    if atlas != None:
        adj_file = os.path.join(filesystem.get_anatomydir(), 'misc', '%s_labels_adj.joblib' % atlas)        
        if os.path.exists(adj_file):
            computeAdj = False
            ladj = joblib.load(adj_file)
        else:
            computeAdj = True
        
    if computeAdj:
    
        # Sourcespace acjacency
        sadj = mne.spatial_src_adjacency(src).todense()
        
        # Number of labels
        P = len(labels)
        
        # Init adjacency
        ladj = np.zeros((P,P))
        
        # Cycle trough all labels on rows
        for i in range(P):        
            v1 = labels[i].get_vertices_used() + (labels[i].hemi == 'rh') * 10242
            for j in range(P):
                v2 = labels[j].get_vertices_used() + (labels[j].hemi == 'rh') * 10242
                
                ladj[i,j] = np.any(sadj[v1,:][:,v2]) * 1
        
        # Convert to sparse MNE python format    
        ladj = sp.sparse.coo_array(ladj)
        
        # Save
        joblib.dump(ladj, adj_file, compress=3)
        
    
    return ladj
        
    
    
    
    
    
    
    
    
    
    

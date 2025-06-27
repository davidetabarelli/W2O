# Utilities

import mne;
import numpy as np;
import scipy as sp
from functools import partial
import os
import matplotlib 
import matplotlib.pyplot as plt
import fabric
import paramiko
import joblib
import datetime

from w2o import utils
from w2o import filesystem
from w2o import sources



#### FUNCTIONS

def get_permutation_number(N=None, tail=0):
    if N == None:
        return 100000
    else:
     return 2 ** (N - (tail == 0))

def prepare_script_for_1w_rm_ANOVA(unique_str):
    
    sc_str =  ""
    sc_str += "\n"
    sc_str += "import numpy as np\n"
    sc_str += "import mne\n"
    sc_str += "import joblib\n"
    sc_str += "\n"
    sc_str += "data_to = joblib.load('w2o_remote_tmp_data_to_%s.joblib')\n" % unique_str
    sc_str += "\n"
    sc_str += "N = len(data_to['X'][0])\n"
    sc_str += "C = len(data_to['X'])\n"
    sc_str += "\n"
    sc_str += "def stat_fun(*args):\n"
    sc_str += "\treturn mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[C], effects='A', return_pvals=False)[0]\n"
    sc_str += "\n"
    sc_str += "thr = mne.stats.f_threshold_mway_rm(N, [C], 'A', data_to['alpha'])\n"
    sc_str += "\n"
    sc_str += "F, cl, clp, _ = mne.stats.permutation_cluster_test(data_to['X'], threshold=thr, n_permutations=data_to['permutations'], tail=1, stat_fun=stat_fun, adjacency=data_to['adj'], n_jobs=140, seed=19579, out_type='mask', buffer_size=None)\n"
    sc_str += "\n"
    sc_str += "sig_cl = np.argwhere(clp <= data_to['alpha']).reshape(-1)\n"
    sc_str += "\n"
    sc_str += "res = {'F': F, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}\n"
    sc_str += "\n"
    sc_str += "\n"
    sc_str += "data_from = {'res': res}\n"
    sc_str += "joblib.dump(data_from, './w2o_remote_tmp_data_from_%s.joblib', compress=3)\n" % unique_str
    sc_str += "\n"
    sc_str += "quit()\n"
    
    return sc_str

def prepare_script_for_1_samp_t_test(unique_str):
    
    sc_str =  ""
    sc_str += "\n"
    sc_str += "import numpy as np\n"
    sc_str += "import scipy as sp\n"
    sc_str += "import mne\n"
    sc_str += "import joblib\n"
    sc_str += "\n"
    sc_str += "data_to = joblib.load('w2o_remote_tmp_data_to_%s.joblib')\n" % unique_str
    sc_str += "\n"
    sc_str += "N = len(data_to['X'][0])\n"
    sc_str += "\n"
    sc_str += "thr = sp.stats.t.ppf(1 - data_to['alpha']/(2 if data_to['tail'] == 0 else 1), N-1)\n"
    sc_str += "\n"
    sc_str += "T, cl, clp, _ = mne.stats.permutation_cluster_1samp_test(data_to['X'], threshold=thr, n_permutations=data_to['permutations'], tail=data_to['tail'], stat_fun=mne.stats.ttest_1samp_no_p, adjacency=data_to['adj'], n_jobs=140, seed=19579, out_type='mask', buffer_size=None)\n"
    sc_str += "\n"
    sc_str += "sig_cl = np.argwhere(clp <= data_to['alpha']).reshape(-1)\n"
    sc_str += "\n"
    sc_str += "res = {'T': T, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}\n"
    sc_str += "\n"
    sc_str += "\n"
    sc_str += "data_from = {'res': res}\n"
    sc_str += "joblib.dump(data_from, './w2o_remote_tmp_data_from_%s.joblib', compress=3)\n" % unique_str
    sc_str += "\n"
    sc_str += "quit()\n"
    
    return sc_str
    

# Compare spectra pooled - T-Test
# IN: spectra = ( (N,F) , (N,F))
def pooled_spectra_1_samp_t_test(spectra, alpha=0.05, tail=0, permutations=get_permutation_number(), irbio_num=3):
    
    # Create stat input
    if type(spectra[0]) == list:
        X = np.asarray(spectra[0]) - np.asarray(spectra[1])
    else:
        X = spectra[0] - spectra[1]
    
    # Adjacency
    adj = mne.stats.combine_adjacency(X.shape[1])
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha, 'tail': tail}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1_samp_t_test(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res
    
# Compare spectra spatially - T-Test
# IN: spectra = ( [AverageSpectrumObjects] , [AverageSpectrumObjects])
def spatial_spectra_1_samp_t_test(spectra, alpha=0.05, tail=0, permutations=get_permutation_number(), irbio_num=3):
    
    # Get sample size
    N = len(spectra[0])
    
    # Prepare stat data
    X = np.transpose(np.asarray([spectra[0][s].get_data() - spectra[1][s].get_data() for s in range(N)]), (0,2,1))
    
    # Adjacency
    sadj, ch_names = mne.channels.find_ch_adjacency(spectra[0][0].info, 'eeg')
    adj = mne.stats.combine_adjacency(len(spectra[0][0].freqs), sadj)
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha, 'tail': tail}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1_samp_t_test(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    
    return res

# Compare spectra bands - T-Test
def fbands_spectra_1_samp_t_test(spectra, info, alpha=0.05, tail=0, permutations=get_permutation_number(), irbio_num=3):
    
    # Get sample size
    N = len(spectra[0])
    
    # Prepare stat data
    X = np.asarray([spectra[0][s] - spectra[1][s] for s in range(N)])
        
    # Adjacency
    adj, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')        
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha, 'tail': tail}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1_samp_t_test(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)    
    
    return res

# Compare spectra spatially - F-Test
# labels can be list of labels or name of atlas
def labels_spectra_1_samp_t_test(lspectra, labels='aparc_sub', alpha=0.05, tail=0, irbio_num=3, permutations=get_permutation_number()):
    
    # Check labels
    if len(labels) == 448:
        labels = 'aparc_sub'
    
    # Prepare stat data
    X = np.transpose(np.asarray([lspectra[0][s] - lspectra[1][s] for s in range(len(lspectra[0]))]), (0,2,1))
    
    # Adjacency
    src = mne.read_source_spaces(os.path.join(filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    if type(labels) == str:
        ladj = sources.get_labels_adjacency(src, labels, labels)
    else:
        ladj = sources.get_labels_adjacency(src, labels)
    
    adj = mne.stats.combine_adjacency(X.shape[1], ladj)
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha, 'tail': tail}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1_samp_t_test(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res

# Compare spectra pooled - F-Test
def pooled_spectra_1w_rm_ANOVA(spectra, alpha=0.05, permutations=get_permutation_number(), irbio_num=3):
    
    # Create stat input
    if type(spectra[0]) == list:
        X = [np.asarray(sp) for sp in spectra]    
    
    # Adjacency
    adj = mne.stats.combine_adjacency(X[0].shape[1])
        
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1w_rm_ANOVA(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res


# Compare spectra spatially - F-Test
def spatial_spectra_1w_rm_ANOVA(spectra, alpha=0.05, permutations=get_permutation_number(), irbio_num=3):
    
    # Prepare stat data
    X = [np.transpose(np.asarray([sp.get_data() for sp in spectra[i]]), (0,2,1)) for i in range(len(spectra))]
        
    # Adjacency
    sadj, ch_names = mne.channels.find_ch_adjacency(spectra[0][0].info, 'eeg')
    adj = mne.stats.combine_adjacency(len(spectra[0][0].freqs), sadj)
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1w_rm_ANOVA(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res



# Compare spectra bands - F-Test
def fbands_spectra_1w_rm_ANOVA(spectra, info, alpha=0.05, permutations=get_permutation_number(), irbio_num=3):
    
    if type(spectra[0]) == list:
        X = [np.asarray(sp) for sp in spectra]
    
    # Adjacency
    adj, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')   
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1w_rm_ANOVA(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res


# Compare spectra spatially - F-Test
# labels can be list of labels or name of atlas
def labels_spectra_1w_rm_ANOVA(lspectra, labels='aparc_sub', alpha=0.05, irbio_num=3, permutations=get_permutation_number()):
    
    # Check labels
    if len(labels) == 448:
        labels = 'aparc_sub'
    
    # Prepare stat data
    X = [np.transpose(np.asarray([sp for sp in lspectra[i]]), (0,2,1)) for i in range(len(lspectra))]
    
    # Adjacency
    src = mne.read_source_spaces(os.path.join(filesystem.get_anatomydir(), 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    if type(labels) == str:
        ladj = sources.get_labels_adjacency(src, labels, labels)
    else:
        ladj = sources.get_labels_adjacency(src, labels)
    
    adj = mne.stats.combine_adjacency(X[0].shape[1], ladj)
    
    # Unique string for files
    now = datetime.datetime.now(); 
    unique_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    
    # Remote files names
    script_file = 'w2o_remote_tmp_script_%s.py' % unique_str
    data_to_file = 'w2o_remote_tmp_data_to_%s.joblib' % unique_str
    data_from_file = 'w2o_remote_tmp_data_from_%s.joblib' % unique_str
    
    # Open remote connection
    conn = filesystem.open_remote_connection(irbio_num)
    
    # Prepare data
    data_to = {'X': X, 'adj': adj, 'permutations': permutations, 'alpha': alpha}
    
    # Save data
    joblib.dump(data_to, os.path.join(filesystem.get_local_wd(), data_to_file), compress=3)
    
    # Prepare and save script    
    with open(os.path.join(filesystem.get_local_wd(), script_file), "w") as f:
        f.write(prepare_script_for_1w_rm_ANOVA(unique_str))
        
    # Transfer data and script
    conn.put(os.path.join(filesystem.get_local_wd(), script_file), filesystem.get_remote_wd())
    conn.put(os.path.join(filesystem.get_local_wd(), data_to_file), filesystem.get_remote_wd())
    
    # Run remote command
    conn.run('conda deactivate; conda activate mne; cd %s; python3 %s' %  (filesystem.get_remote_wd(), script_file))
    
    # Transfer back results
    conn.get(os.path.join(filesystem.get_remote_wd(), data_from_file), os.path.join(filesystem.get_local_wd(), data_from_file))
    
    # Read results
    data_from = joblib.load(os.path.join(filesystem.get_local_wd(), data_from_file))
    res = data_from['res']
        
    # Delete files and variables
    del data_to
    del data_from
    os.remove(os.path.join(filesystem.get_local_wd(), script_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_to_file))
    os.remove(os.path.join(filesystem.get_local_wd(), data_from_file))
        
    # Close connection
    filesystem.close_remote_connection(conn)
    
    return res


    
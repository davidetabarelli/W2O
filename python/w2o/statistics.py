# Utilities

import mne;
import numpy as np;
import scipy as sp
from functools import partial
import matplotlib 
import matplotlib.pyplot as plt

from w2o import utils



#### FUNCTIONS

def get_permutation_number():
    return 100000
    #return 2**ArithmeticError# Exact with N = 22

# Compare spectra pooled - T-Test
# IN: spectra = ( (N,F) , (N,F))
def pooled_spectra_1_samp_t_test(spectra, alpha=0.05, tail=0, permutations=get_permutation_number()):
    
    # Create stat input
    if type(spectra[0]) == list:
        X = np.asarray(spectra[0]) - np.asarray(spectra[1])
    else:
        X = spectra[0] - spectra[1]
    
    # Sample size
    N = X.shape[0]
    
    # Statistical threshold
    thr = sp.stats.t.ppf(1 - alpha / 2, N-1)
    adj = mne.stats.combine_adjacency(X.shape[1])

    T, cl, clp, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=permutations, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'T': T, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res
    
# Compare spectra spatially - T-Test
# IN: spectra = ( [AverageSpectrumObjects] , [AverageSpectrumObjects])
def spatial_spectra_1_samp_t_test(spectra, alpha=0.05, tail=0, permutations=get_permutation_number()):
    
    # Get sample size
    N = len(spectra[0])
    
    # Prepare stat data
    X = np.transpose(np.asarray([spectra[0][s].get_data() - spectra[1][s].get_data() for s in range(N)]), (0,2,1))
    
    # Statistical threshold
    thr = sp.stats.t.ppf(1 - alpha / 2, N-1)
    
    # Adjacency
    sadj, ch_names = mne.channels.find_ch_adjacency(spectra[0][0].info, 'eeg')
    adj = mne.stats.combine_adjacency(len(spectra[0][0].freqs), sadj)
    
    T, cl, clp, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=permutations, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'T': T, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res

# Compare spectra bands - T-Test
def fbands_spectra_1_samp_t_test(spectra, info, alpha=0.05, tail=0, permutations=get_permutation_number()):
    
    # Get sample size
    N = len(spectra[0])
    
    # Prepare stat data
    X = np.asarray([spectra[0][s] - spectra[1][s] for s in range(N)])
    
    # Statistical threshold
    thr = sp.stats.t.ppf(1 - alpha / 2, N-1)
    
    # Adjacency
    adj, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')        
    
    T, cl, clp, _ = mne.stats.permutation_cluster_1samp_test(X, threshold=thr, n_permutations=permutations, tail=tail, stat_fun=mne.stats.ttest_1samp_no_p, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'T': T, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res

# Compare spectra pooled - F-Test
def pooled_spectra_1w_rm_ANOVA(spectra, alpha=0.05, tail=0, permutations=get_permutation_number()):
    
    # Create stat input
    if type(spectra[0]) == list:
        X = [np.asarray(sp) for sp in spectra]    
    
    # Sample size
    N = X[0].shape[0]
    
    # Conditions
    C = len(X)
    
    # Degrees
    #dfn = C - 1
    #dfd = N - C
    
    # Adjacency
    adj = mne.stats.combine_adjacency(X[0].shape[1])
    
    # One-way repeated-measures ANOVA
    def stat_fun(*args):
        # get f-values only.
        return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[C], effects='A', return_pvals=False)[0]
    
    # Statistical threshold
    #thr = sp.stats.f.ppf(1 - alpha, dfn=dfn, dfd=dfd)
    thr = mne.stats.f_threshold_mway_rm(N, [C], 'A', alpha)
    

    #F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=mne.stats.f_oneway, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask')
    F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=stat_fun, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'F': F, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res


# Compare spectra spatially - F-Test
def spatial_spectra_1w_rm_ANOVA(spectra, alpha=0.05, tail=0, permutations=get_permutation_number()):
    
    # Prepare stat data
    X = [np.transpose(np.asarray([sp.get_data() for sp in spectra[i]]), (0,2,1)) for i in range(len(spectra))]
    
    # Get sample size
    N = len(X[0])
    
    # Conditions
    C = len(X)
    
    # Degrees
    #dfn = C - 1
    #dfd = N - C
    
    # Adjacency
    sadj, ch_names = mne.channels.find_ch_adjacency(spectra[0][0].info, 'eeg')
    adj = mne.stats.combine_adjacency(len(spectra[0][0].freqs), sadj)
    
    # One-way repeated-measures ANOVA
    def stat_fun(*args):
        # get f-values only.
        return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[C], effects='A', return_pvals=False)[0]
    
    # Statistical threshold
    #thr = sp.stats.f.ppf(1 - alpha, dfn=dfn, dfd=dfd)
    thr = mne.stats.f_threshold_mway_rm(N, [C], 'A', alpha)
    
    # F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=mne.stats.f_oneway, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask')
    F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=stat_fun, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'F': F, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res



# Compare spectra bands - F-Test
def fbands_spectra_1w_rm_ANOVA(spectra, info, alpha=0.05, permutations=get_permutation_number()):
    
    if type(spectra[0]) == list:
        X = [np.asarray(sp) for sp in spectra]
    
    # Sample size
    N = X[0].shape[0]
    
    # Conditions
    C = len(X)
    
    # Degrees
    #dfn = C - 1
    #dfd = N - C
    
    # Adjacency
    adj, ch_names = mne.channels.find_ch_adjacency(info, 'eeg')   
    
    # One-way repeated-measures ANOVA
    def stat_fun(*args):
        # get f-values only.
        return mne.stats.f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=[C], effects='A', return_pvals=False)[0]
    
    # Statistical threshold
    #thr = sp.stats.f.ppf(1 - alpha, dfn=dfn, dfd=dfd)
    thr = mne.stats.f_threshold_mway_rm(N, [C], 'A', alpha)
    
    #F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=mne.stats.f_oneway, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask')
    F, cl, clp, _ = mne.stats.permutation_cluster_test(X, threshold=thr, n_permutations=permutations, tail=1, stat_fun=stat_fun, adjacency=adj, n_jobs=utils.get_njobs(), seed=19579, out_type='mask', buffer_size=None)
    
    sig_cl = np.argwhere(clp <= alpha).reshape(-1)
    
    res = {'F': F, 'cl': cl, 'clp': clp, 'sig_cl': sig_cl}
    
    return res
    
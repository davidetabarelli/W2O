import os


# Paths
basedir = os.path.dirname(os.path.dirname(os.getcwd()))
datadir = os.path.join(basedir, 'data')
resultsdir = os.path.join(basedir, 'results')

########## FUNCTIONS ##########

def get_basedir():
    
    return basedir


def get_datadir():
    
    return datadir

def get_eegrawsubjectdir(subject):
    
    eegrawsubjectdir = os.path.join(get_datadir(), 'raw', subject)
    
    return eegrawsubjectdir


def get_artfctsubjectdir(subject):
    
    artfctsubjectdir = os.path.join(get_datadir(), 'artifacts', subject)
    if not os.path.exists(artfctsubjectdir):
        os.makedirs(artfctsubjectdir)        

    return artfctsubjectdir




def get_resultssubjectdir(project, subject):
    
    resultssubjectdir = os.path.join(resultsdir, project, subject)
    
    if not os.path.exists(resultssubjectdir):
        os.makedirs(resultssubjectdir)        

    return resultssubjectdir


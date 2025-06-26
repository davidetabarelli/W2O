import os
import fabric

# Paths
basedir = os.path.dirname(os.path.dirname(os.getcwd()))
datadir = os.path.join(basedir, 'data')
anatomydir = os.path.join(basedir, 'data', 'anatomy')
resultsdir = os.path.join(basedir, 'results')

########## FUNCTIONS ##########

def get_basedir():
    
    return basedir


def get_datadir():
    
    return datadir

def get_anatomydir():
    
    return anatomydir

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

def open_remote_connection(irbio_server_num=3):
    
    conn = fabric.Connection("davide.tabarelli@irbio-server%d.cimec.unitn.it" % irbio_server_num, connect_kwargs={'password': "Filmoor1995!"})
    conn.open()
    
    return conn

def close_remote_connection(conn):
    conn.close()
    
def get_remote_wd():
    return "/home/davide.tabarelli/w2o_wd/"

def get_local_wd():
    return os.path.join(get_datadir(), "w2o_wd/")
    

    
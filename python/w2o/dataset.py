import os
import numpy as np
import pandas as pd
import mne

from w2o import filesystem
from w2o import preliminary


##### FUNCTIONS ####

# Subjectr list (static)

def get_subjects():
    
    subjects = ['AROS',
				'EDON',	
				'ELOD',	
				'ERAB',					
				'GNAI',	
				'GNON',	
				'ICIN',					
				'LEBR',	
				'LERA',	
				'LIUA',	
				'LOOM',	
				'MMEN',	
				'NGIC',	
				'NIAL',						
				'NNHE',	
				'NNRA',	
				'OFIE',	
				'RAIO',	
				'RIAT',	
				'SOIS',	
				'TEIP',	
				'URER'#,
                #'ERER'
                # 'IUOR', Troppi buchi
                # 'NNAS', Bughi qua e la. Rivedi
			];
    
    return subjects, len(subjects)


# Data frame version of subject list
def get_dataset():
    
    dataset = pd.Dataframe()
    
    # TODO
    
    return dataset



def get_event_dict():
    
    evt_dict = {}
    
    evt_dict['Instructions_on'] = 100
    evt_dict['Instructions_off'] = 101

    evt_dict['Sound_on'] = 200

    evt_dict['Vib_test_on'] = 22
    evt_dict['Vib_test_off'] = 23
    evt_dict['Vib_1_on'] = 40
    evt_dict['Vib_1_off'] = 41
    evt_dict['Vib_2_on'] = 60
    evt_dict['Vib_2_off'] = 61

    evt_dict['Fix_start'] = 10
    evt_dict['EC_start'] = 20
    evt_dict['Muscles_start'] = 24
    evt_dict['Breath_1_start'] = 30
    evt_dict['Porn_start'] = 50
    evt_dict['Porn_end'] = 51
    evt_dict['Masturbation_start'] = 70
    evt_dict['Pleateau'] = 71
    evt_dict['Orgasm_peak'] = 72
    evt_dict['Orgasm_end'] = 73
    evt_dict['Resolution'] = 75
    evt_dict['Breath_2_start'] = 80

    evt_dict['Routine_end'] = 999
    
    evt_dict['Exp_begin'] = 800
    evt_dict['Exp_end'] = 801
    
    return evt_dict

def get_periods_definition():
    
    periods_def = {}
    
    periods_def['FixRest'] = {'evt_1': 'Fix_start', 'evt_2': 'Routine_end'}
    periods_def['EcRest'] = {'evt_1': 'EC_start', 'evt_2': 'Routine_end'}
    periods_def['VibTest'] = {'evt_1': 'Vib_test_on', 'evt_2': 'Routine_end'}
    periods_def['Muscles'] = {'evt_1': 'Muscles_start', 'evt_2': 'Sound_on'}
    periods_def['Breathe1'] = {'evt_1': 'Breath_1_start', 'evt_2': 'Routine_end'}
    periods_def['Vib1'] = {'evt_1': 'Vib_1_on', 'evt_2': 'Routine_end'}
    periods_def['Porn'] = {'evt_1': 'Porn_start', 'evt_2': 'Porn_end'}
    periods_def['Vib2'] = {'evt_1': 'Vib_2_on', 'evt_2': 'Routine_end'}
    periods_def['Masturbation'] = {'evt_1': 'Masturbation_start', 'evt_2': 'Pleateau'}
    periods_def['Pleateau'] = {'evt_1': 'Pleateau', 'evt_2': 'Orgasm_peak'}
    periods_def['Orgasm'] = {'evt_1': 'Orgasm_peak', 'evt_2': 'Orgasm_end'}
    periods_def['Resolution'] = {'evt_1': 'Orgasm_end', 'evt_2': 'Resolution'}   # Ask for triggers/definition
    periods_def['Breathe2'] = {'evt_1': 'Breath_2_start', 'evt_2': 'Routine_end'}
    
    return periods_def



def get_fbands_dict(mode='standard'):
    
    fbands_dict = {}
    
    # fbands_dict['Delta'] = [2, 4]    
    # fbands_dict['Theta'] = [4, 7]
    # fbands_dict['Alpha'] = [8, 13]
    # fbands_dict['Beta'] = [15, 30]
    # fbands_dict['Gamma'] = [31, 48] 
    
    if mode == 'standard':
        fbands_dict['Delta'] = [2, 4]    
        fbands_dict['Theta'] = [4, 7]
        fbands_dict['Alpha'] = [8, 12]
        fbands_dict['Beta'] = [13, 30]
        fbands_dict['Gamma'] = [31, 40] 
    
    if mode == 'data_driven':
        fbands_dict['Delta'] = [2, 4]    
        fbands_dict['Theta'] = [4, 7]
        fbands_dict['Alpha'] = [8, 13]
        fbands_dict['Beta'] = [16, 27]
        fbands_dict['Gamma'] = [31, 44]         
        
    
    return fbands_dict


def get_dataset_info():
    
    dinfo_file = os.path.join(filesystem.get_resultssubjectdir('preprocessing', 'group'), 'dataset_info.pkl')
    
    if os.path.exists(dinfo_file):
        
        dinfo = pd.read_pickle(dinfo_file)        
        
    else:
        
        subjects, N = get_subjects()
        periods = get_periods_definition().keys()
        
        dinfo = pd.DataFrame({ ** { 
                                    'Subject': pd.Series(dtype='str'), 
                                    'Bad_Channels': pd.Series(dtype='int'), 
                                    'Total_Comps': pd.Series(dtype='int'), 
                                    'Rejected_Comps': pd.Series(dtype='int')
                                },
                              ** {'%s_Time' % period :pd.Series(dtype='float') for period in periods}                      
                            })
        
        for subject in subjects:
            
            nrow = {}
            
            nrow['Subject'] = subject
            
            bads = np.loadtxt(fname=os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-badchannels.txt' % subject), dtype='bytes').astype(str).tolist();
            if bads == []:
                nrow['Bad_Channels'] = 0
            elif type(bads) is list:
                nrow['Bad_Channels'] = 1
            else:
                nrow['Bad_Channels'] = len(bads)
            
            ica = mne.preprocessing.read_ica(os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica.fif' % subject))
            
            nrow['Total_Comps'] = ica.n_components_
            
            ica_excluded = np.loadtxt(fname=os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-excluded.txt' % subject), dtype='bytes').astype(int).tolist();
            ica_enhc_excluded = np.loadtxt(fname=os.path.join(filesystem.get_artfctsubjectdir(subject), '%s-ica-enhc-excluded.txt' % subject), dtype='bytes').astype(int).tolist();
            
            nrow['Rejected_Comps'] = len(ica_excluded + ica_enhc_excluded)
            
            craw, events, evt_dict = preliminary.get_clean_data(subject, True)
            p_raws = preliminary.extract_periods(craw, events, evt_dict, 0.0)
            
            for period in periods:
                nrow['%s_Time' % period] = (p_raws[period].last_samp - p_raws[period].first_samp) / p_raws[period].info['sfreq']
                
            dinfo = pd.concat([dinfo, pd.DataFrame([nrow])], ignore_index=True)
        
        dinfo.to_pickle(dinfo_file)
        
    return dinfo

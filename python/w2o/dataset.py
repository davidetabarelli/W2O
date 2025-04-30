import os
import numpy as np
import pandas as pd
import mne

from w2o import filesystem


##### FUNCTIONS ####

# Subjectr list (static)

def get_subjects():
    
    subjects = ['AROS',
				'EDON',	
				'ELOD',	
				'ERAB',	
				'ERER',	
				'GNAI',	
				'GNON',	
				'ICIN',	
				'IUOR',	
				'LEBR',	
				'LERA',	
				'LIUA',	
				'LOOM',	
				'MMEN',	
				'NGIC',	
				'NIAL',	
				'NNAS',	
				'NNHE',	
				'NNRA',	
				'OFIE',	
				'RAIO',	
				'RIAT',	
				'SOIS',	
				'TEIP',	
				'URER'
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
    
    return evt_dict
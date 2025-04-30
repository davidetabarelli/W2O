addpath('~/Documents/MATLAB/eeglab/')

subjects = {	'AROS',	
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
			};

eeglab 

for i = 1:length(subjects)

	subject = subjects{i};
	
	xdf_file = sprintf('~/W2O/data/raw/%s/%s_eeg.xdf', subject, subject);
	bva_file = sprintf('~/W2O/data/raw/%s/%s_eeg.', subject, subject);

	% Load EEG data
	EEG = pop_loadxdf(xdf_file, 'streamtype', 'EEG', 'exclude_markerstreams', {});
	EEG.setname = 'raw';

	% Fix
	EEG = pop_select(EEG, 'nochannel', 65:67);
	correct_labels = {	'Fp1', 'Fz', 'F3', 'F7', 'LL', 'FC5', 'FC1', 'C3', 'T7', 'HEOG_left', 'CP5', ...
						'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'HEOG_right', 'CP6', 'CP2', 'Cz', 'C4', 'T8', ...
						'VEOG_right', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', ...
						'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', ...
						'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'RA'};


	chanlocs = fullfile('/Users/davide/Documents/MATLAB/eeglab', 'plugins', 'dipfit', 'standard_BESA', 'standard-10-5-cap385.elp');
	for j = 1:length(EEG.chanlocs)
		EEG.chanlocs(j).labels = correct_labels{j};
	end


	% Write
	pop_writebva(EEG, bva_file);

end


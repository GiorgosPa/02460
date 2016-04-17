codedir=strsplit(mfilename('fullpath'),filesep);
disp(['Adding ' fullfile(codedir{1:end-1}) ' to paths']);
addpath(fullfile(codedir{1:end-1}));

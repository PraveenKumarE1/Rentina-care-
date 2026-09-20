function startup()
%STARTUP Configure paths and defaults for the DR screening system.
root = fileparts(mfilename('fullpath'));
addpath(genpath(root));
if exist(fullfile(root,'config','settings.json'),'file')
    fprintf('DR Screening System ready: %s\n', root);
else
    warning('Configuration file is missing: %s', fullfile(root,'config','settings.json'));
end
end

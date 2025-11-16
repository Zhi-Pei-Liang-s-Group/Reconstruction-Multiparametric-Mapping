%% ========================================================================
% script0_setDataPathLibrary.m
% -------------------------------------------------------------------------
% script to setup data path and libraty on SPICE
% =========================================================================

clc;
close all;
fprintf('==== Script run: setup data path and library ==== \n');

%% reset the default path setup 
restoredefaultpath;
set(0,'defaultfigureColor','w');
posFigDefault = [100,100,800,600];
set(0,'defaultfigureposition',posFigDefault);

%% setup the data path

% home path
homePath    = '/home/';
procDatPath = './';
dataSavPath = fullfile(procDatPath,'data',filesep);
niiDataPath = fullfile(dataSavPath,'niiData',filesep);

% figure save path 
figSavePath = fullfile(procDatPath,'figs',filesep);

if(~exist(dataSavPath,'dir'))
    mkdir(dataSavPath);
end
if(~exist(niiDataPath,'dir'))
    mkdir(niiDataPath);
end
if(~exist(figSavePath,'dir'))
    mkdir(figSavePath);
end

% display path information
fprintf(' - Process data path: %s\n',procDatPath);

% display completion message
disp(' - Setup data path complete');

%% suppress some known warning
warning off MATLAB:mir_warning_maybe_uninitialized_temporary
warning off parallel:convenience:RunningJobsExist

%% display completion message
disp(' - Setup libraty complete');
close all;




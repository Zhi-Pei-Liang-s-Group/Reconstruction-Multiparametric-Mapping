% This is the packaged code for multi-parametric water reconstruction for simultaneous MRI and MRSI
% 
% [Inputs]
%   --- supporting data
%       --- tvecEPSI:            time grid along spectroscopic dimension        
%       --- dt_D2:               echo space for EPSI trajectory 
%       --- unippm:              conversion ratio between Hz and ppm
%       --- brainMask:           segmentation mask for the brain
%       --- lipMask:             segmentation mask for the subcutaneous lipid
%       --- gapMask:             segmentation mask for the layer between brain and subcutaneous lipid
%       --- anatRef:             reference MRI image (e.g., MPRAGE)
%
%   --- reconstructed water signals from Ernst angle (from "dense central segment" and "sparse peripheral segment")
%       --- sxt_high_all:        reconstructed water signals from Ernst angle
%       --- sense_map_high:      high-resolution sensitivity map matching 'sxt_high_all'
%
%   --- raw multi-parametric water data
%       --- ktEPSIt1t2x:         raw data for multi-parametric signals
%       --- FAvec:               vector of acquired flip angles
%       --- TPvec:               vector of acquired T2 preparation times
%
% [Outputs]
%   sxt_recon_final:    final reconstruction
%
% 
%   Liang's Lab @ UIUC
%   Created: 2025-11-03
%

clear all;
restoredefaultpath;
close all;
clc
fprintf('==== Script run: reconstruct multi-parametric water signals from "relaxation encoding segment" ==== \n');

%% for debug
dbstop if error

%% ============== step #0: set up paths ============= %%

%% path set up
disp('==== Setup library and path ====');tSet = tic;
% set default lib
script0_setDataPathLibrary;
supportPath     = fullfile('./support/');
addpath(fullfile(supportPath,'Water_T1T2/code/'));
addpath(fullfile(supportPath,'Water_T1T2/module'));
addpath(fullfile(supportPath,'pipeline_function/'));
addpath(fullfile(supportPath,'Utilies/'));
addpath(fullfile(supportPath,'Utilies/genImagePatch/'));
addpath(genpath(fullfile(supportPath,'share_codes/')));
% set data path
prepEPSIfile    = fullfile(dataSavPath,'preparedData.mat');
saveFigFile     = fullfile(procDatPath,'/Figs/multiparametric_water_recon/');
disp(['==== Setup library and path finished, elapsed time: ',num2str(toc(tSet)),' sec ====']);

%% ============== step #1: data loading ============= %%

%% ==== load data info, anatomical reference, and masks ==== %%
disp('---Loading data info, anatomical reference, and masks---');
myLoadVars(prepEPSIfile,'tvecEPSI','dt_D2','unippm','brainMask','lipMask','gapMask','anatRef','reg_mask_arrays','alTR_seconds',0);
anatRef_high    = myLoadVars(prepEPSIfile,'anatRef',0);
brainMask_high  = myLoadVars(prepEPSIfile,'brainMask',0);
lipMask_high    = myLoadVars(prepEPSIfile,'lipMask',0);
reg_mask_arrays = myLoadVars(prepEPSIfile,'reg_mask_arrays',0);

%% ==== load data acquired with the Ernst angle ==== %% 
disp('---Loading reference data (acquired with Ernst angle)---');
sxt_high_all        = myLoadVars(prepEPSIfile,'sxt_high_all',0);
sense_map_high      = myLoadVars(prepEPSIfile,'sense_map_high',0);

%% ==== load T1/T2 data ==== %% 
disp('---Loading raw multiparametric data---');
% raw data load
ktEPSIt1t2x         = myLoadVars(prepEPSIfile,'ktEPSIt1t2x',0);

% set up variables 
dt_all              = dt_D2/2;
[Ny,Nx,Nz,Nt,~]     = size(sxt_high_all);
tvec_all            = tvecEPSI(1:Nt);
Nc                  = size(sense_map_high,5);

% truncate time point
if length(tvecEPSI) > Nt
    for ifm = 1:size(ktEPSIt1t2x,2)
        for ix = 1:Nx
            ktEPSI   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,[],Nc);
            ktEPSIt1t2x{ix,ifm} = vec(ktEPSI(:,:,:,1:Nt,:));
        end
    end
end

%% ==== mask set up ==== %%  
disp('---Set up masks---');
brainMask           = imresize3d(brainMask,[Ny,Nx,Nz]);
lipMask             = imresize3d(lipMask,[Ny,Nx,Nz]);
gapMask             = imfill3D(lipMask,'holes') & ~brainMask & ~lipMask;
anatRef             = imresize3d(anatRef,[Ny,Nx,Nz]);
reg_mask_arrays     = imresize4d(reg_mask_arrays,[Ny,Nx,Nz]);
csfMask             = reg_mask_arrays(:,:,:,4) > 0.1 & brainMask;
wmMask              = reg_mask_arrays(:,:,:,3) > 0.5 & ~csfMask & brainMask;
gmMask              = ~wmMask & ~csfMask & brainMask;
gm_seg              = reg_mask_arrays(:,:,:,2);
wm_seg              = reg_mask_arrays(:,:,:,3);
csf_seg             = reg_mask_arrays(:,:,:,4);
gap_seg             = reg_mask_arrays(:,:,:,5);
lip_seg             = reg_mask_arrays(:,:,:,6);

%% ==== other acquisition parameters ==== %%  
FAvec               = [12,17,22,32];             %@@ flip angles
T2pvec              = [0,0.02,0.04,0.06,0.08];   %@@ T2 preparation times 

%% ============== step #2: multiparametric water reconstruction ============= %%

%% prepare the input

% parse supporting data
support_data                = struct;
support_data.dt_D2          = dt_D2;
support_data.alTR_seconds   = alTR_seconds;
support_data.tvecEPSI       = tvecEPSI;
support_data.FAvec          = FAvec;
support_data.T2pvec         = T2pvec;
support_data.sense_map_high = sense_map_high;
support_data.brainMask      = brainMask;
support_data.lipMask        = lipMask;
support_data.gapMask        = gapMask;
support_data.wmMask         = wmMask;
support_data.gmMask         = gmMask;
support_data.csfMask        = csfMask;
support_data.anatRef        = anatRef;
support_data.gm_seg         = gm_seg;
support_data.wm_seg         = wm_seg;
support_data.csf_seg        = csf_seg;
support_data.lip_seg        = lip_seg;
support_data.gap_seg        = gap_seg;

%% reconstruction process
[sxt_recon,T1map,PDmap,t2map] = function_multiparametric_reconstruction(ktEPSIt1t2x,sxt_high_all,support_data);
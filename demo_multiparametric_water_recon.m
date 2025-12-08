% This is the packaged code for multi-parametric reconstruction for multiplexed MRI
%
% The code contains the following components:
%   --- multiparametric image reconstruction from highly sparse (k,t)-data
%   --- deep learning (DL)-based image super-resolution
%   --- DL-assisted generalized series model-based reconstruction
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
%   T1map_final: high-resolution T1 map
%   PDmap_final: high-resolution PD map
%   T2map_final: high-resolution T2 map
%
% 
%   Liang's Lab @ UIUC
%   Created: 2025-11-03
%

clear all;
close all;
home;
restoredefaultpath;
fprintf('==== Script run: reconstruct multi-parametric water signals from "relaxation encoding segment" ==== \n');

%% for debug
dbstop if error

%% ============== step #0: set up paths ============= %%

%% path set up
disp('==== Setup library and path ====');tSet = tic;

% set default lib
homePath    = '/home/';
procDatPath = './';

supportPath     = fullfile('./support/');
addpath(fullfile(supportPath,'Water_T1T2/code/'));
addpath(fullfile(supportPath,'pipeline_function/'));
addpath(fullfile(supportPath,'Utilies/'));
addpath(genpath(fullfile(supportPath,'share_codes/')));

% set data path
dataSavPath     = fullfile(procDatPath,'data',filesep);
prepEPSIfile    = fullfile(dataSavPath,'preparedData.mat');
saveFigFile     = fullfile(procDatPath,'/Figs/multiparametric_water_recon/');

% check data file
if ~exist(prepEPSIfile,'file')
    error('File %s not found. Please download the data as instructed in README.md and copy it into %s',prepEPSIfile,dataSavPath);
end

disp(['==== Setup library and path finished, elapsed time: ',num2str(toc(tSet)),' sec ====']);

%% ============== step #1: data loading ============= %%

%% load data info, anatomical reference, and masks
disp('---Loading data info, anatomical reference, and masks---');
myLoadVars(prepEPSIfile,'tvecEPSI','dt_D2','unippm','brainMask','lipMask','gapMask','anatRef','reg_mask_arrays','alTR_seconds','adFlipAngle_degree',0);
anatRef_high    = myLoadVars(prepEPSIfile,'anatRef',0);
brainMask_high  = myLoadVars(prepEPSIfile,'brainMask',0);
lipMask_high    = myLoadVars(prepEPSIfile,'lipMask',0);
reg_mask_arrays = myLoadVars(prepEPSIfile,'reg_mask_arrays',0);

%% load data acquired with the Ernst angle
disp('---Loading reference data (acquired with Ernst angle)---');
sxt_high_all        = myLoadVars(prepEPSIfile,'sxt_high_all',0);
sense_map_high      = myLoadVars(prepEPSIfile,'sense_map_high',0);

%% load T1/T2 data
disp('---Loading raw multiparametric data---');
% raw data load
ktEPSIt1t2x         = myLoadVars(prepEPSIfile,'ktEPSIt1t2x',0);
FAvec               = myLoadVars(prepEPSIfile,'FAvec',0);
T2pvec              = myLoadVars(prepEPSIfile,'T2pvec',0);

% pre-trained reconstruction parameters
myLoadVars(prepEPSIfile,'weight_fun','lambdaRecon','lambda_GS','lambda_ref','conv_sz','Rank_gps','noise_sigma',0);

% set up variables 
[Ny,Nx,Nz,Nt,~]     = size(sxt_high_all);

%% mask set up
disp('---Set up masks---');
brainMask           = imresize3d(brainMask,[Ny,Nx,Nz]);
lipMask             = imresize3d(lipMask,[Ny,Nx,Nz]);
gapMask             = imfill3D(lipMask,'holes') & ~brainMask & ~lipMask;
anatRef             = imresize3d(anatRef,[Ny,Nx,Nz]);
reg_mask_arrays     = imresize4d(reg_mask_arrays,[Ny,Nx,Nz]);
csf_seg             = reg_mask_arrays(:,:,:,4);
gap_seg             = reg_mask_arrays(:,:,:,5);
lip_seg             = reg_mask_arrays(:,:,:,6);

%% ============== step #2: multiparametric reconstruction from sparse data ============= %%

% parse supporting data
support_data                    = struct;
support_data.dt_D2              = dt_D2;
support_data.alTR_seconds       = alTR_seconds;
support_data.adFlipAngle_degree = adFlipAngle_degree;
support_data.tvecEPSI           = tvecEPSI;
support_data.FAvec              = FAvec;
support_data.T2pvec             = T2pvec;
support_data.weight_fun         = weight_fun;
support_data.lambdaRecon        = lambdaRecon;
support_data.lambda_GS          = lambda_GS;
support_data.lambda_ref         = lambda_ref;
support_data.conv_sz            = conv_sz;
support_data.Rank_gps           = Rank_gps;
support_data.noise_sigma        = noise_sigma;
support_data.sense_map_high     = sense_map_high;
support_data.brainMask          = brainMask;
support_data.lipMask            = lipMask;
support_data.gapMask            = gapMask;
support_data.anatRef            = anatRef;
support_data.csf_seg            = csf_seg;
support_data.lip_seg            = lip_seg;
support_data.gap_seg            = gap_seg;

% reconstruction process
[sxt_recon,T1map_low,PDmap_low,T2map_low] = function_multiparametric_reconstruction(ktEPSIt1t2x,sxt_high_all,support_data);
image_low = sxt_recon(:,:,:,1,:);
image_low(isnan(image_low)) = 0;

%% ============== step #3: deep learning-based super-resolution ============= %%

%% data preprocessing

%% network prediction - output: T1map_NN, PDmap_NN, T2map_NN


%% ============== step #4: GS-based spatial adaptation ============= %%
disp('==== GS-based spatial adaptation ====');tRecon = tic;

% GS-based reconstruction
[T1map_final,PDmap_final,T2map_final] = function_GS_reconstruction(image_low,T1map_low,PDmap_low,T2map_low,T1map_NN,PDmap_NN,T2map_NN,support_data);

disp(['==== GS-based spatial adaptation finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);


%% ============== step #5: display results ============= %%
figure;montagesc(flip(PDmap_final(:,:,21:6:end-20)));caxis([0,1e-3]);colormap('gray');axis off;
figure;montagesc(flip(T2map_final(:,:,21:6:end-20)));caxis([0,0.3]);colormap('hot');axis off;
figure;montagesc(flip(T1map_final(:,:,21:6:end-20)));caxis([0.,3.5]);colormap('jet');axis off;




function [T1map_final,PDmap_final,T2map_final] = function_GS_reconstruction(image_low,T1map_low,PDmap_low,T2map_low,T1map_NN,PDmap_NN,T2map_NN,support_data)
%% ============== step #0: parse input ============= %%

% parse supporting data
dt_D2               = support_data.dt_D2;
alTR_seconds        = support_data.alTR_seconds;
adFlipAngle_degree  = support_data.adFlipAngle_degree;
FAvec               = support_data.FAvec;
T2pvec              = support_data.T2pvec;
brainMask           = support_data.brainMask;
lipMask             = support_data.lipMask;
gapMask             = support_data.gapMask;
anatRef             = support_data.anatRef;
csf_seg             = support_data.csf_seg;

% set up variables 
dt_all              = dt_D2/2;
[Nyh,Nxh,Nzh]       = size(T1map_NN);
[Ny,Nx,Nz]          = size(T1map_low);

ref_frame           = 1;
tar_frame           = [ref_frame+1:ref_frame+length(FAvec)+length(T2pvec)-1];
T1_frame            = [ref_frame+1:ref_frame+length(FAvec)];
T2_frame            = [ref_frame,ref_frame+length(FAvec)+1:ref_frame+length(FAvec)+length(T2pvec)-1];

%% ============== step #1: pre-processing ============= %%

%% remove NAN
T1map_NN(isnan(T1map_NN)) = 0;
PDmap_NN(isnan(PDmap_NN)) = 0;
T2map_NN(isnan(T2map_NN)) = 0;

%% resize images
brainMask_high      = imresize3d(brainMask,size(T1map_NN));
T1map_2mm_rsz       = imresize3d(T1map_low,size(T1map_NN));
PDmap_2mm_rsz       = imresize3d(PDmap_low,size(T1map_NN));
T2map_2mm_rsz       = imresize3d(T2map_low,size(T1map_NN));

%% outlier removal

% T1 map
T1map_2mm_rsz(T1map_2mm_rsz>3.5) = 3.5;
T1map_2mm_rsz(T1map_2mm_rsz<0)   = 0;
T1map_NN(T1map_NN>3.5)           = 3.5;
T1map_NN(T1map_NN<0)             = 0;

% PD map
PDmap_2mm_rsz(PDmap_2mm_rsz>3e-3)= 3e-3;
PDmap_2mm_rsz(PDmap_2mm_rsz<0)   = 0;
PDmap_NN(PDmap_NN>3e-3)          = 3e-3;
PDmap_NN(PDmap_NN<0)             = 0;

% T2 map
T2map_2mm_rsz(T2map_2mm_rsz>0.3) = 0.3;
T2map_2mm_rsz(T2map_2mm_rsz<0)   = 0;
T2map_NN(T2map_NN>0.3)           = 0.3;
T2map_NN(T2map_NN<0)             = 0;

% low-resolution image
for index_fm = 1:size(image_low,4)
    image_low(:,:,:,index_fm) = despike_local_hampel(image_low(:,:,:,index_fm), [3,3,3], 100, imresize3d(brainMask,[Ny,Nx,Nz]));
end
image_low(isnan(image_low)) = 0;

%% ============== step #2: GS compensation on biomarker maps ============= %%

%% T1 map
conv_sz      = [96,96,1]; 
recon_opt    = struct('maxit',50);
T1map_high2  = T1map_NN;
parfor index_z = 1:Nzh
    img_ref       = T1map_NN(:,:,index_z);
    img_tar       = T1map_2mm_rsz(:,:,index_z);
    Mask_fit      = brainMask_high(:,:,index_z);
    [h,img_recon] = GSrecon3D_fftBased_fast_windowed(F3_x2k(applySpatialSupport3d(img_tar,Mask_fit)),F3_x2k(applySpatialSupport3d(img_ref,Mask_fit)),conv_sz,recon_opt);
    T1map_high2(:,:,index_z) = real(img_recon);
end
T1map_high2  = T1map_high2.*brainMask_high;

%% PD map
conv_sz      = [96,96,1]; 
recon_opt    = struct('maxit',50);
PDmap_high2  = PDmap_NN;
parfor index_z = 1:Nzh
    img_ref      = PDmap_NN(:,:,index_z);
    img_tar      = PDmap_2mm_rsz(:,:,index_z);
    Mask_fit     = brainMask_high(:,:,index_z);
    [h,img_recon] = GSrecon3D_fftBased_fast_windowed(F3_x2k(applySpatialSupport3d(img_tar,Mask_fit)),F3_x2k(applySpatialSupport3d(img_ref,Mask_fit)),conv_sz,recon_opt);
    PDmap_high2(:,:,index_z) = real(img_recon);
end
PDmap_high2  = PDmap_high2.*brainMask_high;

%% T2 map
conv_sz      = [96,96,1]; 
recon_opt    = struct('maxit',50);
T2map_high2  = T2map_NN;
parfor index_z = 1:Nzh
    img_ref      = T2map_NN(:,:,index_z);
    img_tar      = T2map_2mm_rsz(:,:,index_z);
    Mask_fit     = brainMask_high(:,:,index_z);
    [h,img_recon] = GSrecon3D_fftBased_fast_windowed(F3_x2k(applySpatialSupport3d(img_tar,Mask_fit)),F3_x2k(applySpatialSupport3d(img_ref,Mask_fit)),conv_sz,recon_opt);
    T2map_high2(:,:,index_z) = real(img_recon);
end
T2map_high2  = T2map_high2.*brainMask_high;


%% ============== step #3: Image synthesize via biophysical model ============= %%

%% synthesize T2 maps

% T1 maps

% variables set up
FAvec2      = [adFlipAngle_degree,FAvec]; 
TRepsi2     = alTR_seconds;

% VFA images synthesis
conc_map_T1 = synFlashVFAfromT1PDMap(T1map_high2,PDmap_high2,FAvec2,TRepsi2,brainMask_high);
conc_map_T1(isnan(conc_map_T1)) = 0;

%% synthesize T2 maps

% variables set up
FA          = adFlipAngle_degree;
T2prep      = T2pvec;

% VFA images synthesis
conc_map_T2 = synT2prepGREfromM0T2Map(T2map_high2,PDmap_high2,T1map_high2,T2prep,FA,TRepsi2,brainMask_high);
conc_map_T2(isnan(conc_map_T2)) = 0;

%% combined concentration maps
conc_map_syn_NN = cat(4,conc_map_T1,conc_map_T2(:,:,:,2:end));
for index_fm = 1:size(conc_map_syn_NN,4)
    conc_map_syn_NN(:,:,:,index_fm) = despike_local_hampel(conc_map_syn_NN(:,:,:,index_fm), [3,3,3], 100, brainMask_high);
end


%% ============== step #4: GS compensation on images ============= %%

%% GS-based spatial adaptation
GS_order            = [24,Nx,Nz];
[conc_map_syn_GS,~] = fastGSrecon3D_parforEcho(squeeze(applySpatialSupport3d(image_low,brainMask)),conc_map_syn_NN,struct('Mask_fit',brainMask,'order',GS_order));


%% ============== step #5: final parameter fitting ============= %%

%% T1 fitting 
sxt_fit             = abs(conc_map_syn_GS(:,:,:,T1_frame));
[T1map_final,PDmap_final] = estT1map3DfromFlashVFA(sxt_fit,support_data.FAvec,support_data.alTR_seconds,brainMask);

%% PD from EPSI
sxt_epsi            = abs(conc_map_syn_GS(:,:,:,ref_frame));
PDmap_epsi          = estPDmap3DfromSPICEwithT1(sxt_epsi,T1map_final,support_data.dt_D2/2,support_data.adFlipAngle_degree,support_data.alTR_seconds,brainMask);
PDmap_final         = imresize3d(csf_seg,[Nyh,Nxh,Nzh]).*PDmap_final + (1-imresize3d(csf_seg,[Nyh,Nxh,Nzh])).*PDmap_epsi;

%% T2 fitting 
sxt_fit             = abs(conc_map_syn_GS(:,:,:,T2_frame));
[T2map_final,~]     = estT2map3DfromT2prepGRE_multiEcho(sxt_fit,T1map_final,PDmap_final,support_data.T2pvec,support_data.adFlipAngle_degree,support_data.alTR_seconds,brainMask);


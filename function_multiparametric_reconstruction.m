function [sxt_recon,T1map,PDmap,t2map] = function_multiparametric_reconstruction(ktEPSIt1t2x,sxt_high_all,support_data,flag_tissuePrior,flag_lastIter)

%% ============== step #0: parse input ============= %%

% parse supporting data
dt_D2               = support_data.dt_D2;
alTR_seconds        = support_data.alTR_seconds;
tvecEPSI            = support_data.tvecEPSI;
FAvec               = support_data.FAvec;
T2pvec              = support_data.T2pvec;
sense_map_high      = support_data.sense_map_high;
brainMask           = support_data.brainMask;
lipMask             = support_data.lipMask;
gapMask             = support_data.gapMask;
wmMask              = support_data.wmMask;
gmMask              = support_data.gmMask;
csfMask             = support_data.csfMask;
anatRef             = support_data.anatRef;
gm_seg              = support_data.gm_seg;
wm_seg              = support_data.wm_seg;
csf_seg             = support_data.csf_seg;
lip_seg             = support_data.lip_seg;
gap_seg             = support_data.gap_seg;

if varsNotexistOrIsempty('flag_tissuePrior')
    flag_tissuePrior = false; % defaul is to turn off the tissue-based GS model
end

if varsNotexistOrIsempty('flag_lastIter')
    flag_lastIter    = false;    % defaul is to turn off last iteration
end

% set up variables 
dt_all              = dt_D2/2;
[Ny,Nx,Nz,Nt,~]     = size(sxt_high_all);
tvec_all            = tvecEPSI(1:Nt);
Nc                  = size(sense_map_high,5);

%% ============== step #1: reconstruction, temporal GS model ============= %%
disp('==== Image reconstruction, iter #1 ====');tRecon = tic;

%% set variables
Nfm                = size(ktEPSIt1t2x,2);

%% image recon    
s_xt_recon3D = zeros(Ny,Nx,Nz,Nt,Nfm,'single');
Ind_ix = 1;
pLen   = 0;
for ix = 1:Nx
    
    pLen = printProgress(pLen,ix,Nx,' - Recon-slice: ');
    
    %% select slices
    brainMask_slc  = brainMask(:,ix,:,:);
    lipMask_slc    = lipMask(:,ix,:,:);
    gapMask_slc    = gapMask(:,ix,:,:);
    sxt_ref        = sxt_high_all(:,ix,:,:);
    anatRef_slc    = anatRef(:,ix,:,:);
    smapSlc        = sense_map_high(:,ix,:,:,:);
    
    if sum(brainMask_slc(:)|lipMask_slc(:)|gapMask_slc(:)) == 0
        s_xt_recon3D(:,ix,:,:,:) = zeros(Ny,1,Nz,Nt,Nfm,'single');
        Ind_ix = Ind_ix + 1;
        continue;
    end

    %% weighting function (FID)
    weight_T2  = 50e-3;
    weight_fun = (vec(exp(-gen_tvec(Nt,dt_all)/weight_T2)));

    %% edge weights
    Dg         = genWL2mtx(permute(anatRef_slc,[1,3,2]),struct('ratio',30));
    
    %% GS reconstruction (frame by frame)
    Mask         = brainMask_slc|lipMask_slc|gapMask_slc;
    optRecon   	 = struct('lambda',9e-10,'maxit',30,'cgtol',1e-8,'disp',false);
    sxt_recon_all = zeros(Ny,1,Nz,Nt,Nfm,'like',sxt_high_all);
    parfor ifm = 1:Nfm
        % prepare k-space data
        ktEPSI   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,Nt,Nc);
        ktMask   = abs(ktEPSI(:,:,:,:,1))>1e-10;
        % add weighting (weighted least square)
        ktEPSI   = bsxfun(@times,ktEPSI,permute(weight_fun,[2,3,4,1]));
        sxt_refw = bsxfun(@times,sxt_ref,permute(weight_fun,[2,3,4,1]));
        % GS based reconstruction
        sxtRecon = sense_T1T2_recon_wl2(ktEPSI,sxt_refw,ktMask,Mask,smapSlc,Dg,optRecon);
        sxt_recon_all(:,:,:,:,ifm) = sxtRecon;
    end

    %% record results
    s_xt_recon3D(:,ix,:,:,:) = sxt_recon_all;
    Ind_ix = Ind_ix + 1;

end
s_xt_recon3D = cat(5,sxt_high_all,s_xt_recon3D);
disp(['==== Image reconstruction, iter #1 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);


%% ============== step #2: reconstruction, spatial GS model ============= %%
disp('==== Image reconstruction, iter #2 ====');tRecon = tic;

%% ===== obtain spatial constraint via spatial GS ===== %%

if ~flag_tissuePrior
    
    %% GS parameters
    conv_sz      = [64,64,64]; % 64 matches cnumber of encodings encodings
    conv_sz_cell = {conv_sz;conv_sz;conv_sz;};
    
    %% tissue masks
    Mask_cell    = {brainMask;gapMask;lipMask};
    NofRegion    = length(Mask_cell);
    
    %% reference images
    img_ref      = sumOfSqr(s_xt_recon3D(:,:,:,:,1),4);
    
    %% GS recon
    recon_opt    = struct('maxit',50);
    conc_map_GS  = zeros(Ny,Nx,Nz,1,Nfm,'single');
    parfor ifm = 1:Nfm
        img_tar      = s_xt_recon3D(:,:,:,1,1+ifm);
        img_recon    = 0;
        for index_roi = 1:NofRegion
            [h,img_recon_tmp] = GSrecon3D_fftBased_fast(F3_x2k(applySpatialSupport3d(img_tar,Mask_cell{index_roi})),F3_x2k(applySpatialSupport3d(img_ref,Mask_cell{index_roi})),conv_sz_cell{index_roi},recon_opt);
            img_recon = img_recon + img_recon_tmp;
        end
        conc_map_GS(:,:,:,:,ifm) = img_recon;
    end
    
else
    
    %% GS parameters
    conv_sz      = [16,16,16]; % reduced order as tissue-based
    conv_sz_cell = {conv_sz;conv_sz;conv_sz;conv_sz;conv_sz};
    
    %% tissue masks
    Mask_cell    = {gmMask;wmMask;csfMask;gapMask;lipMask};
    NofRegion    = length(Mask_cell);
    
    %% reference images
    img_ref      = sumOfSqr(s_xt_recon3D(:,:,:,:,1),4);
    
    %% GS recon
    recon_opt    = struct('maxit',50);
    conc_map_GS  = zeros(Ny,Nx,Nz,1,Nfm,'single');
    parfor ifm = 1:Nfm
        img_tar      = s_xt_recon3D(:,:,:,1,1+ifm);
        img_recon    = 0;
        for index_roi = 1:NofRegion
            [h,img_recon_tmp] = GSrecon3D_fftBased_fast(F3_x2k(applySpatialSupport3d(img_tar,Mask_cell{index_roi})),F3_x2k(applySpatialSupport3d(img_ref,Mask_cell{index_roi})),conv_sz_cell{index_roi},recon_opt);
            img_recon = img_recon + img_recon_tmp;
        end
        conc_map_GS(:,:,:,:,ifm) = img_recon;
    end
    
end

%% ===== reperform reconstruction with constraint on concentration ===== %%
s_xt_recon3D_iter2 = zeros(Ny,Nx,Nz,Nt,Nfm,'single');
Ind_ix = 1;
pLen   = 0;
for ix = 1:Nx
    
%     disp(['Processing slice ',num2str(ix)]);
    pLen = printProgress(pLen,ix,Nx,' - Recon-slice: ');
    
    Nfm            = size(ktEPSIt1t2x,2);

    %% select slices
    brainMask_slc  = brainMask(:,ix,:,:);
    lipMask_slc    = lipMask(:,ix,:,:);
    gapMask_slc    = gapMask(:,ix,:,:);
    wmMask_slc     = wmMask(:,ix,:,:);
    gmMask_slc     = gmMask(:,ix,:,:);
    csfMask_slc    = csfMask(:,ix,:,:);
    anatRef_slc    = anatRef(:,ix,:,:);
    sxt_ref        = sxt_high_all(:,ix,:,:);
    smapSlc        = sense_map_high(:,ix,:,:,:);
    conc_map_slc   = conc_map_GS(:,ix,:,:,:);
    
    if sum(brainMask_slc(:)|lipMask_slc(:)|gapMask_slc(:)) == 0
        s_xt_recon3D_iter2(:,ix,:,:,:) = zeros(Ny,1,Nz,Nt,Nfm,'single');
        Ind_ix = Ind_ix + 1;
        continue;
    end

    %% weighting function (FID)
    weight_T2  = 50e-3;
    weight_fun = (vec(exp(-gen_tvec(Nt,dt_all)/weight_T2)));

    %% GS reconstruction (frame by frame)
%     disp('--- GS model reconstruction ---');tSet = tic;
    lambda_ref   = 0.015;
    conv_sz      = [Ny,1,Nz];
    conv_sz_cell = {conv_sz};
    Mask_cell    = {brainMask_slc|lipMask_slc|gapMask_slc};
    optRecon   	 = struct('maxit',30,'cgtol',1e-8,'disp',false);
    sxt_recon_all = zeros(Ny,1,Nz,Nt,Nfm,'like',sxt_high_all);
    parfor ifm = 1:Nfm
%         printProgress2(ifm,Nfm);
        % prepare k-space data
        ktEPSI   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,Nt,Nc);
        ktMask   = abs(ktEPSI(:,:,:,:,1))>1e-10;
        % add weighting (weighted least square)
        ktEPSI   = bsxfun(@times,ktEPSI,permute(weight_fun,[2,3,4,1]));
        sxt_refw = bsxfun(@times,sxt_ref,permute(weight_fun,[2,3,4,1]));
        % concentration reference
        conc_ref = conc_map_slc(:,:,:,:,ifm);
        % GS based reconstruction
        sxtRecon = sense_spatialGS_recon_regional_wRef(ktEPSI,sxt_refw,ktMask,conv_sz_cell,Mask_cell,smapSlc,conc_ref,lambda_ref,optRecon);
        sxt_recon_all(:,:,:,:,ifm) = sxtRecon;
    end
%     disp(['--- GS model reconstruction finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

    %% record results
    s_xt_recon3D_iter2(:,ix,:,:,:) = sxt_recon_all;
    Ind_ix = Ind_ix + 1;

end
s_xt_recon3D_iter2 = cat(5,sxt_high_all,s_xt_recon3D_iter2);

disp(['==== Image reconstruction, iter #2 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);

%% fit T1 maps (3D)

num_frame   = 4;  
T1_frame    = [2,3,4,5];

disp('--- T1 map fitting ---');tSet = tic;
% T1 fitting 
NtimeFit    = 20;
TRepsi      = alTR_seconds;
ind         = 1;
sxt_fit     = zeros([Ny,Nx,Nz,NtimeFit,num_frame]);
for ifm = T1_frame
    sxt_fit(:,:,:,:,ind)     = s_xt_recon3D_iter2(:,:,:,1:NtimeFit,ifm);
    ind = ind+1;
end
sxt_fit     = abs(permute(sxt_fit,[1,2,3,5,4]));
FAvec2      = [FAvec];
[T1map_LR,PDmap_tmp,sxt_fit] = estT1map3DfromFlashVFA(sxt_fit,FAvec2,TRepsi,brainMask|lipMask|gapMask);
T1map_LR(T1map_LR>4) = 4;
T1map_LR(T1map_LR<0) = 4;
PDmap_tmp   = abs(PDmap_tmp(:,:,:,1));

% PD from EPSI
sxt_epsi    = s_xt_recon3D_iter2(:,:,:,1:30,1);
FA          = 27;
PDmap_epsi  = estPDmap3DfromSPICEwithT1(sxt_epsi,T1map_LR,dt_D2/2,FA,TRepsi,brainMask|lipMask|gapMask);
PDmap       = csf_seg.*PDmap_tmp + (1-csf_seg).*PDmap_epsi;
PDmap       = brainMask.*PDmap + ~brainMask.*PDmap_tmp;

disp(['--- T1 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

%% fit T2 maps (3D)

num_frame   = 5;
T2_frame    = [1,6,7,8,9];

disp('--- T2 map fitting ---');tSet = tic;
% T1 fitting 
NtimeFit    = 1;
TRepsi      = alTR_seconds;
ind         = 1;
sxt_fit     = zeros([Ny,Nx,Nz,NtimeFit,num_frame]);
for ifm = T2_frame
    sxt_fit(:,:,:,:,ind)     = s_xt_recon3D_iter2(:,:,:,1:NtimeFit,ifm);
    ind = ind+1;
end
sxt_fit       = abs(permute(sxt_fit,[1,2,3,5,4]));
FA            = 27;
T2prep        = [0,0.02,0.04,0.06,0.08];
[t2map,pdmap] = estT2map3DfromT2prepGRE_multiEcho(sxt_fit,T1map_LR,PDmap,T2prep,FA,TRepsi,brainMask|lipMask|gapMask);
M0map         = pdmap;
disp(['--- T2 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

%% ============== step #3: reconstruction, subspace prior (in probabilistic sense) ============= %%
disp('==== Image reconstruction, iter #3 ====');tRecon = tic;

%% ==== synthesize concentration maps from T1, PD, and T2 ==== %%

%% synthesize T1 maps

% variables set up
FAvec2      = [27,FAvec]; % SPICE frame at 1st

% VFA images synthesis
conc_map_T1 = synFlashVFAfromT1PDMap(T1map_LR,PDmap,FAvec2,TRepsi,brainMask|lipMask|gapMask);
conc_map_T1(isnan(conc_map_T1)) = 0;

%% synthesize T2 maps

% variables set up
FA          = 27;
T2prep      = T2pvec(2:end);
TRepsi      = alTR_seconds;

% VFA images synthesis
conc_map_T2 = synT2prepGREfromM0T2Map(t2map,PDmap,T1map_LR,T2prep,FA,TRepsi,brainMask|lipMask|gapMask);
conc_map_T2(isnan(conc_map_T2)) = 0;

%% combined concentration maps
conc_map_syn = cat(4,conc_map_T1,conc_map_T2);

%% ==== determine posteriori distribution based on T1 map ==== %%

%% tissue masks
Mask_cell                   = {gmMask;wmMask;csfMask;lipMask;gapMask};
NofRegion                   = length(Mask_cell);

%% prior distribution (based on segmentation)

% prior distribution - based on segmentation
prior_map                   = cat(4,gm_seg,wm_seg,csf_seg,lip_seg.*(gapMask|lipMask),gap_seg.*(gapMask|lipMask));
prior_map(prior_map>1)      = 1;
prior_map(prior_map<0)      = 0;
prior_map                   = bsxfun(@rdivide,prior_map,sum(prior_map,4));
prior_map(isnan(prior_map)) = 0;
prior_map(isinf(prior_map)) = 0;
prior_map                   = applySpatialSupport3d(prior_map,brainMask|gapMask|lipMask);

% % soften the prior (squeeze to [a,b] to reduce the effect of prior)
a                           = 0.02;
b                           = 0.98;
prior_map                   = prior_map * (b-a);
prior_map                   = prior_map + a;
prior_map(:,:,:,4)          = bsxfun(@times,prior_map(:,:,:,4),lipMask);
prior_map(:,:,:,5)          = bsxfun(@times,prior_map(:,:,:,5),gapMask);
prior_map                   = bsxfun(@rdivide,prior_map,sum(prior_map,4));
prior_map(isnan(prior_map)) = 0;
prior_map(isinf(prior_map)) = 0;
prior_map                   = applySpatialSupport3d(prior_map,brainMask|gapMask|lipMask);

%% likelihood distribution

% estimate the likelihood parameters
num_mix                     = 1; % number of gaussian
likeli_cell                 = cell(NofRegion,1);
for index_mask = 1:NofRegion
    X          = oper3D_mask_4D_data(T1map_LR,Mask_cell{index_mask});
    gm_full    = fitgmdist(X,num_mix);
    likeli_cell{index_mask} = gm_full;
end

% estimate the liklihood map
likeli_map = zeros(Ny,Nx,Nz,NofRegion,'single');
for index_mask = 1:NofRegion
    MU         = [likeli_cell{index_mask}.mu];
    SIGMA      = [likeli_cell{index_mask}.Sigma];
    PPp        = [likeli_cell{index_mask}.PComponents];
    objA       = gmdistribution(MU,SIGMA,PPp);
    likeli_map(:,:,:,index_mask) = oper3D_mask_4D_data_adj(pdf(objA,oper3D_mask_4D_data(T1map_LR,brainMask|gapMask|lipMask)),brainMask|gapMask|lipMask);
end

%% posteriori distribution
prob_map     = likeli_map .* prior_map;
prob_map     = bsxfun(@rdivide,prob_map,sum(prob_map,4));
prob_map(isnan(prob_map)) = 0;
prob_map(isinf(prob_map)) = 0;
prob_map     = applySpatialSupport3d(prob_map,brainMask|gapMask|lipMask);

%% ==== subspace projection (imposing distribution) ==== %%
Rank_gp1      = 6; % gm, wm, csf
Rank_gp3      = 4; % lip
Rank_gp4      = 4; % gap

%% subspace structure and distribution estimation

% subspace estimation - gp1
[~,S,Vt_gp1]  = estMathBases(conc_map_syn,gmMask|wmMask|csfMask);
Vt_gp1        = Vt_gp1(:,1:Rank_gp1);

% distribution estimation - gp1
mu_roi_gp1    = zeros(NofRegion-2,Rank_gp1);
Sigma_roi_gp1 = zeros(Rank_gp1,Rank_gp1,NofRegion-2);
for index_roi = 1:NofRegion-2
    % coefficient samples
    tmp       = reshape(oper3D_mask_4D_data(conc_map_syn,Mask_cell{index_roi}),[],size(conc_map_syn,4));
    [~,coef_samp] = projOntoSubspace(tmp,Vt_gp1,2,1);
    % distribution estimation
    mu_roi_gp1(index_roi,:) = mean(coef_samp,1);
    Sigma_roi_gp1(:,:,index_roi) = diag(var(coef_samp,0,1));
end

% subspace estimation - gp3
[~,S,Vt_gp3]  = estMathBases(conc_map_syn,lipMask);
Vt_gp3        = Vt_gp3(:,1:Rank_gp3);

% distribution estimation - gp3
mu_roi_gp3    = zeros(1,Rank_gp3);
Sigma_roi_gp3 = zeros(Rank_gp3,Rank_gp3,1);
for index_roi = 1:1
    % coefficient samples
    tmp       = reshape(oper3D_mask_4D_data(conc_map_syn,Mask_cell{index_roi+3}),[],size(conc_map_syn,4));
    [~,coef_samp] = projOntoSubspace(tmp,Vt_gp3,2,1);
    % distribution estimation
    mu_roi_gp3(index_roi,:) = mean(coef_samp,1);
    Sigma_roi_gp3(:,:,index_roi) = diag(var(coef_samp,0,1));
end

% subspace estimation - gp4
[~,S,Vt_gp4]  = estMathBases(conc_map_syn,gapMask);
Vt_gp4        = Vt_gp4(:,1:Rank_gp4);

% distribution estimation - gp4
mu_roi_gp4    = zeros(1,Rank_gp4);
Sigma_roi_gp4 = zeros(Rank_gp4,Rank_gp4,1);
for index_roi = 1:1
    % coefficient samples
    tmp       = reshape(oper3D_mask_4D_data(conc_map_syn,Mask_cell{index_roi+4}),[],size(conc_map_syn,4));
    [~,coef_samp] = projOntoSubspace(tmp,Vt_gp4,2,1);
    % distribution estimation
    mu_roi_gp4(index_roi,:) = mean(coef_samp,1);
    Sigma_roi_gp4(:,:,index_roi) = diag(var(coef_samp,0,1));
end

%% subspace projection
Mask_fit         = brainMask|gapMask|lipMask;
options          = SetStructDefault(struct, {'display','Tolx','TolFun','max_iter_mog'}, {'off',1e-5,1e-5,5});   
conc_map_proj    = conc_map_syn;
noise_sigma_gp1  = 1.5e-5; % brain
noise_sigma_gp3  = 4e-5;   % lip
noise_sigma_gp4  = 4e-5;   % gap
% pLen = parfor_progressbar(Ny,'Parfor computing...');
parfor index_y = 1:Ny
%     pLen.iterate(1);
    for index_x = 1:Nx
        for index_z = 1:Nz
            
            if ~Mask_fit(index_y,index_x,index_z)
                continue;
            end
            
            % extract signal
            s_t            = vec(conc_map_syn(index_y,index_x,index_z,:));
            
            if wmMask(index_y,index_x,index_z) || gmMask(index_y,index_x,index_z) || csfMask(index_y,index_x,index_z)
            
                % construct mixture of gaussian
                num_mog_cell   = {3};
                mix_mog_cell   = {vec(prob_map(index_y,index_x,index_z,1:3))};
                mean_mog_cell  = {mu_roi_gp1};
                Sigma_mog_cell = {Sigma_roi_gp1};
                Scale_cell     = {[1;0]};
                
                % subspace projection
                [x_recon,fval_iter] = linear_fitting_map_multi_mog_enforceReal(s_t,Vt_gp1,Rank_gp1,Scale_cell,num_mog_cell,mix_mog_cell,mean_mog_cell,Sigma_mog_cell,noise_sigma_gp1,options);
                
                % synthesize signal
                s_t_fit        = Vt_gp1 * x_recon;
            
            elseif lipMask(index_y,index_x,index_z)
                
                % construct mixture of gaussian
                num_mog_cell   = {1};
                mix_mog_cell   = {vec(prob_map(index_y,index_x,index_z,4))};
                mean_mog_cell  = {mu_roi_gp3};
                Sigma_mog_cell = {Sigma_roi_gp3};
                Scale_cell     = {[1;0]};
                
                % subspace projection
                [x_recon,fval_iter] = linear_fitting_map_multi_mog_enforceReal(s_t,Vt_gp3,Rank_gp3,Scale_cell,num_mog_cell,mix_mog_cell,mean_mog_cell,Sigma_mog_cell,noise_sigma_gp3,options);
            
                % synthesize signal
                s_t_fit        = Vt_gp3 * x_recon;
                
            elseif gapMask(index_y,index_x,index_z)
                
                % construct mixture of gaussian
                num_mog_cell   = {1};
                mix_mog_cell   = {vec(prob_map(index_y,index_x,index_z,5))};
                mean_mog_cell  = {mu_roi_gp4};
                Sigma_mog_cell = {Sigma_roi_gp4};
                Scale_cell     = {[1;0]};
                
                % subspace projection
                [x_recon,fval_iter] = linear_fitting_map_multi_mog_enforceReal(s_t,Vt_gp4,Rank_gp4,Scale_cell,num_mog_cell,mix_mog_cell,mean_mog_cell,Sigma_mog_cell,noise_sigma_gp4,options);
                
                % synthesize signal
                s_t_fit        = Vt_gp4 * x_recon;
                
            end           
            
            % record results
            conc_map_proj(index_y,index_x,index_z,:) = s_t_fit;
            
        end
    end
end
% close(pLen);

% exclude Ernst-angle frame
conc_map_proj   = conc_map_proj(:,:,:,2:end);
conc_map_proj   = permute(conc_map_proj,[1,2,3,5,4]);

% remove outlier pt
for index_fm = 1:size(conc_map_proj,5)
    conc_map_proj(:,:,:,1,index_fm) = despike_local_hampel(conc_map_proj(:,:,:,1,index_fm), [3,3,3], 10, imresize3d(brainMask|lipMask|gapMask,[Ny,Nx,Nz]));
end

% enforce data consistency to center k-space
conc_map_proj_dataCon = conc_map_proj;
conc_map_proj_dataCon(:,:,:,:,1:4) = enforceDataCon(conc_map_proj(:,:,:,:,1:4),imresize_ktrunc(imresize_ktrunc(s_xt_recon3D_iter2(:,:,:,1,2:5),[24,Nx,24],0,0),[Ny,Nx,Nz],0,0),ktrunc(true([24,Nx,24]),[Ny,Nx,Nz]));

if ~flag_lastIter
    
    % synthesize reconstructed signal
    s_xt_recon3D_iter3 = s_xt_recon3D_iter2;
    s_xt_recon3D_iter3(:,:,:,:,2:end) = bsxfun(@times,s_xt_recon3D_iter3(:,:,:,:,2:end),conc_map_proj_dataCon./s_xt_recon3D_iter3(:,:,:,1,2:end));
    
    disp(['==== Image reconstruction, iter #3 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);
    
else
    
    %% ===== reperform reconstruction with constraint on concentration ===== %%
    s_xt_recon3D_iter3 = zeros(Ny,Nx,Nz,Nt,Nfm,'single');
    Ind_ix = 1;
    pLen   = 0;
    for ix = 1:Nx
        
        %     disp(['Processing slice ',num2str(ix)]);
        pLen = printProgress(pLen,ix,Nx,' - Recon-slice: ');
        
        Nfm            = size(ktEPSIt1t2x,2);
        
        %% select slices
        brainMask_slc  = brainMask(:,ix,:,:);
        lipMask_slc    = lipMask(:,ix,:,:);
        gapMask_slc    = gapMask(:,ix,:,:);
        wmMask_slc     = wmMask(:,ix,:,:);
        gmMask_slc     = gmMask(:,ix,:,:);
        csfMask_slc    = csfMask(:,ix,:,:);
        anatRef_slc    = anatRef(:,ix,:,:);
        sxt_ref        = sxt_high_all(:,ix,:,:);
        smapSlc        = sense_map_high(:,ix,:,:,:);
        conc_map_slc   = conc_map_proj_dataCon(:,ix,:,:,:);
        
        if sum(brainMask_slc(:)|lipMask_slc(:)|gapMask_slc(:)) == 0
            s_xt_recon3D_iter3(:,ix,:,:,:) = zeros(Ny,1,Nz,Nt,Nfm,'single');
            Ind_ix = Ind_ix + 1;
            continue;
        end
        
        %% weighting function (FID)
        weight_T2  = 50e-3;
        weight_fun = (vec(exp(-gen_tvec(Nt,dt_all)/weight_T2)));
        
        %% GS reconstruction (frame by frame)
        %     disp('--- GS model reconstruction ---');tSet = tic;
        lambda_ref   = 10;
        conv_sz      = [Ny,1,Nz];
        conv_sz_cell = {conv_sz};
        Mask_cell    = {brainMask_slc|lipMask_slc|gapMask_slc};
        optRecon   	 = struct('maxit',10,'cgtol',1e-8,'disp',false);
        sxt_recon_all = zeros(Ny,1,Nz,Nt,Nfm,'like',sxt_high_all);
        parfor ifm = 1:Nfm
            %         printProgress2(ifm,Nfm);
            % prepare k-space data
            ktEPSI   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,Nt,Nc);
            ktMask   = abs(ktEPSI(:,:,:,:,1))>1e-10;
            % add weighting (weighted least square)
            ktEPSI   = bsxfun(@times,ktEPSI,permute(weight_fun,[2,3,4,1]));
            sxt_refw = bsxfun(@times,sxt_ref,permute(weight_fun,[2,3,4,1]));
            % concentration reference
            conc_ref = conc_map_slc(:,:,:,:,ifm);
            % GS based reconstruction
            sxtRecon = sense_spatialGS_recon_regional_wRef(ktEPSI,sxt_refw,ktMask,conv_sz_cell,Mask_cell,smapSlc,conc_ref,lambda_ref,optRecon);
            sxt_recon_all(:,:,:,:,ifm) = sxtRecon;
        end
        %     disp(['--- GS model reconstruction finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);
        
        %% record results
        s_xt_recon3D_iter3(:,ix,:,:,:) = sxt_recon_all;
        Ind_ix = Ind_ix + 1;
        
    end
    s_xt_recon3D_iter3 = cat(5,sxt_high_all,s_xt_recon3D_iter3);
    
    disp(['==== Image reconstruction, iter #3 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);
    
    
end


%% ============== step #5: parameter quantification ============= %%

disp('==== quantitative map estimation ====');tRecon = tic;
%% fit T1 maps (3D)

num_frame   = 4;  
T1_frame    = [2,3,4,5];

disp('--- T1 map fitting ---');tSet = tic;
% T1 fitting 
NtimeFit    = 1;
TRepsi      = alTR_seconds;
ind         = 1;
sxt_fit     = zeros([Ny,Nx,Nz,NtimeFit,num_frame]);
for ifm = T1_frame
    sxt_fit(:,:,:,:,ind)     = s_xt_recon3D_iter3(:,:,:,1:NtimeFit,ifm);
    ind = ind+1;
end
sxt_fit     = abs(permute(sxt_fit,[1,2,3,5,4]));
FAvec2      = FAvec;
[T1map_iter2,PDmap_tmp2,sxt_fit] = estT1map3DfromFlashVFA(sxt_fit,FAvec2,TRepsi,brainMask);
T1map_iter2(T1map_iter2>5) = 5;
T1map_iter2(T1map_iter2<0) = 5;

% PD from EPSI
sxt_epsi    = s_xt_recon3D_iter3(:,:,:,1:30,1);
FA          = 27;
PDmap_epsi  = estPDmap3DfromSPICEwithT1(sxt_epsi,T1map_iter2,dt_D2/2,FA,TRepsi,brainMask);
PDmap_iter2 = csf_seg.*PDmap_tmp2 + (1-csf_seg).*PDmap_epsi;

disp(['--- T1 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

%% fit T2 maps (3D)

num_frame   = 5;
T2_frame    = [1,6,7,8,9];

disp('--- T2 map fitting ---');tSet = tic;
% T1 fitting 
NtimeFit    = 1;
TRepsi      = alTR_seconds;
ind         = 1;
sxt_fit     = zeros([Ny,Nx,Nz,NtimeFit,num_frame]);
for ifm = T2_frame
    sxt_fit(:,:,:,:,ind)     = s_xt_recon3D_iter3(:,:,:,1:NtimeFit,ifm);
    ind = ind+1;
end
sxt_fit       = abs(permute(sxt_fit,[1,2,3,5,4]));
FA            = 27;
T2prep        = T2pvec;
[t2map_iter2,pdmap] = estT2map3DfromT2prepGRE_multiEcho(sxt_fit,T1map_iter2,PDmap_iter2,T2prep,FA,TRepsi,brainMask);
M0map         = pdmap;
disp(['--- T2 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

disp(['==== quantitative map estimation finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);


%% ============== step #6: rename the output ============= %%
sxt_recon           = s_xt_recon3D_iter3;
T1map               = T1map_iter2;
PDmap               = PDmap_iter2;
t2map               = t2map_iter2;


end


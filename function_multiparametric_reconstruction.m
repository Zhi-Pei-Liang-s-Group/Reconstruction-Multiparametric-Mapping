function [sxt_recon,T1map,PDmap,T2map] = function_multiparametric_reconstruction(ktEPSIt1t2x,sxt_high_all,support_data)
%% ============== step #0: parse input ============= %%

% parse supporting data
dt_D2               = support_data.dt_D2;
alTR_seconds        = support_data.alTR_seconds;
adFlipAngle_degree  = support_data.adFlipAngle_degree;
FAvec               = support_data.FAvec;
T2pvec              = support_data.T2pvec;
weight_fun          = support_data.weight_fun;
lambdaRecon         = support_data.lambdaRecon;
lambda_GS           = support_data.lambda_GS;
lambda_ref          = support_data.lambda_ref;
conv_sz             = support_data.conv_sz;
Rank_gps            = support_data.Rank_gps;
noise_sigma         = support_data.noise_sigma;
sense_map_high      = support_data.sense_map_high;
brainMask           = support_data.brainMask;
lipMask             = support_data.lipMask;
gapMask             = support_data.gapMask;
anatRef             = support_data.anatRef;
csf_seg             = support_data.csf_seg;

% set up variables 
dt_all              = dt_D2/2;
[Ny,Nx,Nz,Nt,~]     = size(sxt_high_all);
Nc                  = size(sense_map_high,5);
Nfm                 = size(ktEPSIt1t2x,2);

ref_frame           = 1;
tar_frame           = [ref_frame+1:ref_frame+length(FAvec)+length(T2pvec)-1];
T1_frame            = [ref_frame+1:ref_frame+length(FAvec)];
T2_frame            = [ref_frame,ref_frame+length(FAvec)+1:ref_frame+length(FAvec)+length(T2pvec)-1];

%% ============== step #1: reconstruction, temporal GS model ============= %%
disp('==== Image reconstruction, iter #1 ====');tRecon = tic;

s_xt_recon3D = zeros(Ny,Nx,Nz,Nt,Nfm,'single');
pLen   = 0;
for ix = 1:Nx
    
    pLen = printProgress(pLen,ix,Nx,' - Recon-slice: ');
    
    % select slices
    brainMask_slc  = brainMask(:,ix,:,:);
    lipMask_slc    = lipMask(:,ix,:,:);
    gapMask_slc    = gapMask(:,ix,:,:);
    sxt_ref        = sxt_high_all(:,ix,:,:);
    anatRef_slc    = anatRef(:,ix,:,:);
    smapSlc        = sense_map_high(:,ix,:,:,:);
    
    if sum(brainMask_slc(:)|lipMask_slc(:)|gapMask_slc(:)) == 0
        s_xt_recon3D(:,ix,:,:,:) = zeros(Ny,1,Nz,Nt,Nfm,'single');
        continue;
    end

    % edge weights
    Dg          = genWL2mtx(permute(anatRef_slc,[1,3,2]));
    
    % temporal GS reconstruction (frame by frame)
    Mask          = brainMask_slc|lipMask_slc|gapMask_slc;
    optRecon   	  = struct('lambda',lambdaRecon,'disp',false);
    sxt_recon_all = zeros(Ny,1,Nz,Nt,Nfm,'like',sxt_high_all);
    parfor ifm = 1:Nfm
        % prepare k-space data
        ktData   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,Nt,Nc);
        ktMask   = abs(ktData(:,:,:,:,1))>1e-10;
        % add weighting (weighted least square)
        ktData   = bsxfun(@times,ktData,weight_fun);
        sxt_refw = bsxfun(@times,sxt_ref,weight_fun);
        % GS based reconstruction
        sxtRecon = sense_T1T2_recon_wl2(ktData,sxt_refw,ktMask,Mask,smapSlc,Dg,optRecon);
        sxt_recon_all(:,:,:,:,ifm) = sxtRecon;
    end

    % record results
    s_xt_recon3D(:,ix,:,:,:) = sxt_recon_all;

end
s_xt_recon3D = cat(5,sxt_high_all,s_xt_recon3D);

disp(['==== Image reconstruction, iter #1 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);

%% ============== step #2: reconstruction, spatial GS model ============= %%
disp('==== Image reconstruction, iter #2 ====');tRecon = tic;

%% obtain spatial constraint for concentration map via spatial GS

% tissue masks
Mask_cell    = {brainMask;gapMask;lipMask};
NofRegion    = length(Mask_cell);

% reference images
img_ref      = sumOfSqr(s_xt_recon3D(:,:,:,:,ref_frame),4);

% GS recon
conc_map_GS  = zeros(Ny,Nx,Nz,1,Nfm,'single');
parfor ifm = 1:Nfm
    for index_x = 1:Nx
        img_tar      = s_xt_recon3D(:,index_x,:,1,1+ifm);
        img_refx     = img_ref(:,index_x,:);
        img_recon    = 0;
        for index_roi = 1:NofRegion
            Mask_fit     = Mask_cell{index_roi}(:,index_x,:);
            if nnz(Mask_fit) == 0
                img_recon_tmp = 0;
            else
                [~,img_recon_tmp] = GSrecon3D_fftBased_fast(F3_x2k(applySpatialSupport3d(img_tar,Mask_fit)),F3_x2k(applySpatialSupport3d(img_refx,Mask_fit)),conv_sz);
            end
            img_recon = img_recon + img_recon_tmp;
        end
        conc_map_GS(:,index_x,:,:,ifm) = img_recon;
    end
end

% re-introduce data, effectively solving argmin_\rho ||d - \rho||^2 + lambda*||\rho_GS - \rho||^2
conc_map_GS = lambda_GS*conc_map_GS + (1-lambda_GS)*s_xt_recon3D(:,:,:,1,tar_frame);

%% reperform reconstruction with constraint on concentration
s_xt_recon3D_iter2 = zeros(Ny,Nx,Nz,Nt,Nfm,'single');
pLen   = 0;
for ix = 1:Nx
    
    pLen = printProgress(pLen,ix,Nx,' - Recon-slice: ');
    
    % select slices
    brainMask_slc  = brainMask(:,ix,:,:);
    lipMask_slc    = lipMask(:,ix,:,:);
    gapMask_slc    = gapMask(:,ix,:,:);
    sxt_ref        = sxt_high_all(:,ix,:,:);
    smapSlc        = sense_map_high(:,ix,:,:,:);
    conc_map_slc   = conc_map_GS(:,ix,:,:,:);
    
    if sum(brainMask_slc(:)|lipMask_slc(:)|gapMask_slc(:)) == 0
        s_xt_recon3D_iter2(:,ix,:,:,:) = zeros(Ny,1,Nz,Nt,Nfm,'single');
        continue;
    end
    
    % GS reconstruction (frame by frame)
    conv_sz      = [Ny,1,Nz];
    conv_sz_cell = {conv_sz};
    Mask_cell    = {brainMask_slc|lipMask_slc|gapMask_slc};
    optRecon   	 = struct('disp',false);
    sxt_recon_all = zeros(Ny,1,Nz,Nt,Nfm,'like',sxt_high_all);
    parfor ifm = 1:Nfm
        % prepare k-space data
        ktData   = reshape(single(full(ktEPSIt1t2x{ix,ifm})),Ny,1,Nz,Nt,Nc);
        ktMask   = abs(ktData(:,:,:,:,1))>1e-10;
        % add weighting (weighted least square)
        ktData   = bsxfun(@times,ktData,weight_fun);
        sxt_refw = bsxfun(@times,sxt_ref,weight_fun);
        % concentration reference
        conc_ref = conc_map_slc(:,:,:,:,ifm);
        % GS based reconstruction
        sxtRecon = sense_spatialGS_recon_regional_wRef(ktData,sxt_refw,ktMask,conv_sz_cell,Mask_cell,smapSlc,conc_ref,lambda_ref,optRecon);
        sxt_recon_all(:,:,:,:,ifm) = sxtRecon;
    end
    
    % record results
    s_xt_recon3D_iter2(:,ix,:,:,:) = sxt_recon_all;

end
s_xt_recon3D_iter2 = cat(5,sxt_high_all,s_xt_recon3D_iter2);

disp(['==== Image reconstruction, iter #2 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);

%% ============== step #3: reconstruction, subspace prior (in probabilistic sense) ============= %%
disp('==== Image reconstruction, iter #3 ====');tRecon = tic;

%% ==== synthesize concentration maps from T1, PD, and T2 ==== %%

% T1 fitting 
sxt_fit     = abs(permute(s_xt_recon3D_iter2(:,:,:,1,T1_frame),[1,2,3,5,4]));
[T1map_LR,PDmap_tmp] = estT1map3DfromFlashVFA(sxt_fit,FAvec,alTR_seconds,brainMask|lipMask|gapMask);
PDmap_tmp   = abs(PDmap_tmp(:,:,:,1));

% PD from EPSI
sxt_epsi    = s_xt_recon3D_iter2(:,:,:,1,ref_frame);
PDmap_epsi  = estPDmap3DfromSPICEwithT1(sxt_epsi,T1map_LR,dt_all,adFlipAngle_degree,alTR_seconds,brainMask|lipMask|gapMask);
PDmap       = csf_seg.*PDmap_tmp + (1-csf_seg).*PDmap_epsi;
PDmap       = brainMask.*PDmap + ~brainMask.*PDmap_tmp;

% T2 fitting 
sxt_fit       = abs(permute(s_xt_recon3D_iter2(:,:,:,1,T2_frame),[1,2,3,5,4]));
[T2map,~]     = estT2map3DfromT2prepGRE_multiEcho(sxt_fit,T1map_LR,PDmap,T2pvec,adFlipAngle_degree,alTR_seconds,brainMask|lipMask|gapMask);

% synthesize T1 maps
conc_map_T1 = synFlashVFAfromT1PDMap(T1map_LR,PDmap,[adFlipAngle_degree,FAvec],alTR_seconds,brainMask|lipMask|gapMask);
conc_map_T1(isnan(conc_map_T1)) = 0;

% synthesize T2 maps
conc_map_T2 = synT2prepGREfromM0T2Map(T2map,PDmap,T1map_LR,T2pvec(2:end),adFlipAngle_degree,alTR_seconds,brainMask|lipMask|gapMask);
conc_map_T2(isnan(conc_map_T2)) = 0;

% combined concentration maps
conc_map_syn = cat(4,conc_map_T1,conc_map_T2);

%% ==== determine posteriori distribution based on T1 map ==== %%

%% tissue masks
Mask_cell     = {brainMask;lipMask;gapMask};
NofRegion     = length(Mask_cell);
prob_map      = cat(4,brainMask,lipMask,gapMask);

%% subspace structure and distribution estimation
Vt_cell       = cell(NofRegion,1);
mu_cell       = cell(NofRegion,1);
Sigma_cell    = cell(NofRegion,1);

for n_region = 1:NofRegion
    % subspace estimation
    [~,~,Vt_tmp]  = estMathBases(conc_map_syn,Mask_cell{n_region});
    Vt_tmp        = Vt_tmp(:,1:Rank_gps(n_region));
    
    % coefficient samples
    tmp           = reshape(oper3D_mask_4D_data(conc_map_syn,Mask_cell{n_region}),[],size(conc_map_syn,4));
    [~,coef_samp] = projOntoSubspace(tmp,Vt_tmp,2,1);
    
    % distribution estimation
    mu_tmp        = mean(double(coef_samp),1);
    Sigma_tmp     = diag(var(double(coef_samp),0,1));
    
    % record
    Vt_cell{n_region} = Vt_tmp;
    mu_cell{n_region} = mu_tmp;
    Sigma_cell{n_region} = Sigma_tmp;
end

%% subspace projection
Mask_fit         = brainMask|gapMask|lipMask;
conc_map_proj    = conc_map_syn;
parfor index_y = 1:Ny
    for index_x = 1:Nx
        for index_z = 1:Nz
            
            if ~Mask_fit(index_y,index_x,index_z)
                continue;
            end
            
            % extract signal
            s_t            = vec(conc_map_syn(index_y,index_x,index_z,:));
            
            for n_region = 1:NofRegion
                if Mask_cell{n_region}(index_y,index_x,index_z)
                    
                    % construct mixture of gaussian
                    num_mog_cell   = {1};
                    mix_mog_cell   = {vec(prob_map(index_y,index_x,index_z,n_region))};
%                     mean_mog_cell  = {mu_cell{n_region}};
%                     Sigma_mog_cell = {Sigma_cell{n_region}};
                    mean_mog_cell  = mu_cell(n_region);
                    Sigma_mog_cell = Sigma_cell(n_region);
                    Scale_cell     = {[1;0]};
                    Vt_recon       = Vt_cell{n_region};
                    Rank_recon     = size(Vt_cell{n_region},2);
                    
                    % coefficient estimation
                    [x_recon,~] = linear_fitting_map_multi_mog_enforceReal(s_t,Vt_recon,Rank_recon,Scale_cell,num_mog_cell,mix_mog_cell,mean_mog_cell,Sigma_mog_cell,noise_sigma);
                    
                    % synthesize signal
                    s_t_fit        = Vt_recon * x_recon;
                    
                    continue;
                end
            end
            
            % record results
            conc_map_proj(index_y,index_x,index_z,:) = s_t_fit;
            
        end
    end
end

% exclude Ernst-angle frame
conc_map_proj   = conc_map_proj(:,:,:,tar_frame);
conc_map_proj   = permute(conc_map_proj,[1,2,3,5,4]);

% remove outlier pt
for index_fm = 1:size(conc_map_proj,5)
    conc_map_proj(:,:,:,1,index_fm) = despike_local_hampel(conc_map_proj(:,:,:,1,index_fm),[],[],imresize3d(brainMask|lipMask|gapMask,[Ny,Nx,Nz]));
end

% synthesize reconstructed signal
s_xt_recon3D_iter3 = s_xt_recon3D_iter2;
s_xt_recon3D_iter3(:,:,:,:,tar_frame)   = bsxfun(@times,s_xt_recon3D_iter3(:,:,:,:,tar_frame),conc_map_proj./s_xt_recon3D_iter3(:,:,:,1,tar_frame));

disp(['==== Image reconstruction, iter #3 finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);

%% ============== step #5: parameter quantification ============= %%

disp('==== quantitative map estimation ====');tRecon = tic;

disp('--- T1 map fitting ---');tSet = tic;
% T1 fitting 
sxt_fit     = abs(permute(s_xt_recon3D_iter3(:,:,:,1,T1_frame),[1,2,3,5,4]));
[T1map_iter2,PDmap_tmp2] = estT1map3DfromFlashVFA(sxt_fit,FAvec,alTR_seconds,brainMask);

% PD from EPSI
sxt_epsi    = s_xt_recon3D_iter3(:,:,:,1,ref_frame);
PDmap_epsi  = estPDmap3DfromSPICEwithT1(sxt_epsi,T1map_iter2,dt_all,adFlipAngle_degree,alTR_seconds,brainMask);
PDmap_iter2 = csf_seg.*PDmap_tmp2 + (1-csf_seg).*PDmap_epsi;

disp(['--- T1 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

disp('--- T2 map fitting ---');tSet = tic;

% T2 fitting 
sxt_fit       = abs(permute(s_xt_recon3D_iter3(:,:,:,1,T2_frame),[1,2,3,5,4]));
[T2map_iter2,~] = estT2map3DfromT2prepGRE_multiEcho(sxt_fit,T1map_iter2,PDmap_iter2,T2pvec,adFlipAngle_degree,alTR_seconds,brainMask);
disp(['--- T2 map fitting finished, elapsed time: ',num2str(toc(tSet)),' sec ---']);

disp(['==== quantitative map estimation finished, elapsed time: ',num2str(toc(tRecon)),' sec ====']);

%% ============== step #6: prepare the output ============= %%
sxt_recon           = s_xt_recon3D_iter3;
T1map               = T1map_iter2;
PDmap               = PDmap_iter2;
T2map               = T2map_iter2;

end

 %#ok<*NBRAK>
 %#ok<*PFBNS>
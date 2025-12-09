function [sxtc_recon,sxtc_recon1] = fastGSrecon3D_parforEcho(sxtc_tar,sxtc_ref,opt)
%% ==================================================================
%FASTGSRECON3D fast GS compensation reconstruction
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2019-06-24
%
%   [INPUTS]
%   ---- Required ----
%   sxtc_tar                low-resolution target (y,x,z,t,coil)
%   sxtc_ref                high-resolution reference (y,x,z,t,coil)
%
%   ---- Optional ----
%   opt
%    - Mask_fit             mask for fitting fast GS [all true]
%    - verbose              degree of verbosity [2]
%
%   [OUTPUTS]
%   sxtc_recon              compensated data (y,x,z,t,coil)
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2019/06/24
%
%   Future: make the code more efficient
%       (e.g. k-space shift, enforce data consistency...)
%
%--------------------------------------------------------------------------

    
    %% parse inputs
    if varsNotexistOrIsempty('opt')
        opt      = struct;
    end
        
    opt = SetStructDefault(opt,{'lambda','order','Mask_fit','verbose'},{1e-7,[],true(size_dims(sxtc_tar,1:3)),2});
    lambda       = opt.lambda;
    order        = opt.order;
    Mask_fit     = opt.Mask_fit;
    verbose      = opt.verbose;
    
    
    %% set up dimensions
    high_res     = size_dims(sxtc_ref,1:3);
    if isempty(order)
        low_res  = size_dims(sxtc_tar,1:3);
    else
        low_res  = order;
        sxtc_tar = imresize_ktrunc(sxtc_tar,low_res,0,0);
    end
    
    if size(sxtc_tar,4)==size(sxtc_ref,4)
        M        = size(sxtc_tar,4);
    else
        warning('Time point mismatch...');
        M        = min(size(sxtc_tar,4),size(sxtc_ref,4));
        sxtc_tar = sxtc_tar(:,:,:,1:M,:);
        sxtc_ref = sxtc_ref(:,:,:,1:M,:);
    end
    
    assert(size(sxtc_tar,5)==size(sxtc_ref,5),'Channel numbers mismatch...');
    Nc           = size(sxtc_tar,5);
    
    %% coil-by-coil, echo-by-echo compensation
    sxtc_recon = initLike(sxtc_ref);
    sxtc_recon1 = initLike(sxtc_ref);
    sampMask   = true(low_res);
    sampMask   = ktrunc(sampMask,high_res);
    
    if(verbose>0)
        tic;
    end
    parfor echo_ind = 1:M
        for coil_ind = 1:Nc
            %% extract the data
            sxt_tar             = sxtc_tar(:,:,:,echo_ind,coil_ind);
            sxt_ref             = sxtc_ref(:,:,:,echo_ind,coil_ind);
            
            %% truncation of reference and zero-padding of both
            sxt_tar_zpad_nofilt = imresize_ktrunc(sxt_tar,high_res,0,0);
            sxt_tar_zpad        = imresize_ktrunc(imresize_ktrunc(sxt_tar,high_res,1,0),high_res,1,0);
            sxt_ref_zpad        = imresize_ktrunc(imresize_ktrunc(imresize_ktrunc(sxt_ref,low_res,0,0),high_res,1,0),high_res,1,0);
            
            %% compensation
            w_fun1              = bsxfun(@rdivide,sxt_tar_zpad,sxt_ref_zpad+eps);
            w_fun1              = applySpatialSupport3d(w_fun1,Mask_fit);
            sxt_ref_fwd         = bsxfun(@times,sxt_ref,w_fun1);
            sxt_recon           = sxt_ref_fwd;
%             sxt_recon           = enforceDataCon(sxt_ref_fwd,sxt_tar_zpad_nofilt,sampMask);
            sxtc_recon1(:,:,:,echo_ind,coil_ind) = sxt_ref_fwd;
            
            %% one more iteration
            w_fun2              = bsxfun(@rdivide,sxt_recon,sxt_ref+eps);
            w_fun2              = applySpatialSupport3d(w_fun2,Mask_fit);
            sxt_ref_fwd2        = bsxfun(@times,sxt_ref,w_fun2);
            sxt_recon           = sxt_ref_fwd2;
%             sxt_recon           = enforceDataCon(sxt_ref_fwd2,sxt_tar_zpad_nofilt,sampMask);
            
            sxtc_recon(:,:,:,echo_ind,coil_ind) = sxt_recon;
        end
    end
    
    if(verbose>0)
        toc;
    end
    
end

% backup functions: fastGSrecon3D_backup_parforEcho.m

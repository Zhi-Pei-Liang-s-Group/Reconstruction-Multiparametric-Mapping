function [h,img_recon] = GSrecon3D_fftBased_fast(d_tar,d_ref,conv_sz,opt)
% this is a fast implementation for FFT-based 3D GS-model reconstruction
%
% Author: Yudu Li
% Date: 2022/08/30
%

opt        = SetStructDefault(opt,{'maxit','cgtol','disp'},...
                                  {    200,  1e-10, false});
maxit      = opt.maxit;
cgtol      = opt.cgtol;
disp       = opt.disp;

%% data size
[L1,L2,L3] = size(d_ref);

%% Fourier recon
x_ref      = F3_k2x(d_ref);
x_tar      = F3_k2x(d_tar);

%% iterative deconvolution

% forward operator
oper_K          = @(x)forward_model(x,x_ref,conv_sz) * sqrt(L1*L2*L3);
oper_K_adj      = @(x)adjoint_model(x,x_ref,conv_sz) * sqrt(L1*L2*L3);

% construct the linear system
linear_sys      = @(x,mode)prep_Afun(x,mode,oper_K,oper_K_adj); %@@ use lsmr

% lsmr
h               = lsmr(linear_sys, x_tar(:), 0, cgtol, cgtol, [], maxit, [], disp); %@@ use lsmr

%% synthesize recon images
img_recon       = reshape(oper_K(h),[L1,L2,L3]);

end

function y = forward_model(x,x_ref,conv_sz)

% data size
[L1,L2,L3] = size(x_ref);

% reshape x
h          = reshape(x,conv_sz);

% zero-padding
h_zp       = ktrunc(h,[L1,L2,L3]);

% fourier transform
h_x_zp     = F3_k2x(h_zp);

% apply filter
y          = x_ref .* h_x_zp;

% vectorize
y          = y(:);

end

function y = adjoint_model(x,x_ref,conv_sz)

% data size
[L1,L2,L3] = size(x_ref);

% reshape x
y          = reshape(x,[L1,L2,L3]);

% apply filter
y          = conj(x_ref) .* y;

% fourier transform
y          = F3_x2k(y);

% k-space truncation
y          = ktrunc(y,conv_sz);

% vectorize
y          = y(:);

end


function y = prep_Afun(x,mode,oper_A,oper_AH)
    switch mode
        case {'notransp',1}
            y = oper_A(x);
        case {'transp',2}
            y = oper_AH(x);
        otherwise
            error('?');
    end
end

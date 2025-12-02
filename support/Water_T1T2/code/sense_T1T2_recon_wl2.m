function sxtRecon = sense_T1T2_recon_wl2(dktMeas,sxtRef,kMask,Mask,senseMap,Dg,opt)

%% program options
opt             = SetStructDefault(opt,{'lambda','maxit','cgtol','disp'},...
                                       {       0,      30,  1e-8, false});
maxit           = opt.maxit;
cgtol           = opt.cgtol;
disp            = opt.disp;
lambda          = opt.lambda;

%% param setup
[Ny,Nx,Nz,Nt]	= size(sxtRef);
Nc              = size(senseMap,5);

%% shift the k-space center
kMaskSh      	= fftshift(fftshift(fftshift(kMask,1),2),3);
dktMeas     	= F3_k2x(dktMeas);
dktMeas       	= F3_x2k(dktMeas,true);

%% iterative deconvolution

% forward operator
oper_K          = @(x)forward_model(x,sxtRef,kMaskSh,Mask,senseMap,Dg,lambda);
oper_K_adj      = @(x)adjoint_model(x,sxtRef,kMaskSh,Mask,senseMap,Dg,lambda);

% check adjoint
if disp
    rel_err     = checkAdjoint(oper_K,oper_K_adj,[Ny*Nx*Nz,1],1e-4,1);
end

% construct the linear system
linear_sys      = @(x,mode)prep_Afun(x,mode,oper_K,oper_K_adj); %@@ use lsmr

% prepare data
d               = [oper4D_mask_4D_data(dktMeas,kMaskSh);zeros(size(Dg,1),1,'single');];

% lsmr
x               = lsmr(linear_sys, d(:), 0, cgtol, cgtol, [], maxit, [], disp); %@@ use lsmr

%% synthesize recon signal

% apply GS-modulation to reference signal
y             = applySpatialSupport3d(reshape(x,[Ny,Nx,Nz]),Mask);
sxtRecon      = bsxfun(@times,sxtRef,y);

end

function y = forward_model(x,sxtRef,kMask,Mask,senseMap,Dg,lambda)

%% data size info
[Ny,Nx,Nz,Nt] = size(kMask);
Nc            = size(senseMap,5);

%% forward pass

% apply spatial mask
y             = applySpatialSupport3d(reshape(x,[Ny,Nx,Nz]),Mask);

% apply GS-modulation to reference signal
y             = bsxfun(@times,sxtRef,y);

% apply sensitivity map
y             = bsxfun(@times,senseMap,y);

% Fourier transform
y             = F3_x2k(y,true);

% sampling mask
y             = oper4D_mask_4D_data(y,kMask);

% vectorize
y             = y(:);

%% regularization
w             = single((Dg*double(x))); 
w             = w(:);

%% combine imaging operator and regularization operator
y  	          = [y;sqrt(lambda)*w];

end

function y = adjoint_model(x,sxtRef,kMask,Mask,senseMap,Dg,lambda)

%% data size info
[Ny,Nx,Nz,Nt] = size(kMask);
Nc            = size(senseMap,5);

%% decompose input
y             = x(1:end-size(Dg,1));
w             = x(end-size(Dg,1)+1:end);

%% adjoint pass

% zero-filling
y             = oper4D_mask_4D_data_adj(y,kMask);

% Fourier transform
y             = F3_k2x(y,true);

% apply sensitivity map
y             = sum(bsxfun(@times,conj(senseMap),y),5);

% apply GS-modulation to reference signal
y             = sum(bsxfun(@times,conj(sxtRef),y),4);

% apply spatial mask
y             = applySpatialSupport3d(y,Mask);

%% adjoint regularization
w             = single((Dg'*double(w)));          % NX*Ngs
w             = w(:);


% output
y             = y(:)+sqrt(lambda)*w;

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

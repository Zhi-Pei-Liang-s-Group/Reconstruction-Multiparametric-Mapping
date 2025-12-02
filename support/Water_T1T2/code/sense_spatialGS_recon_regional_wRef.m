function sxtRecon = sense_spatialGS_recon_regional_wRef(dktMeas,sxtRef,kMask,conv_sz_cell,Mask_cell,senseMap,img_ref,lambda_ref,opt)

%% program options
opt             = SetStructDefault(opt,{'maxit','cgtol','disp'},...
                                       {     30,  1e-10, false});
maxit           = opt.maxit;
cgtol           = opt.cgtol;
disp            = opt.disp;

%% param setup
[Ny,Nx,Nz,Nt]	= size(sxtRef);
Nc              = size(senseMap,5);
NofRegion       = length(Mask_cell);
len             = zeros(NofRegion,1);
for nm = 1 : NofRegion
    len(nm)     = prod(conv_sz_cell{nm});
end

%% shift the k-space center
kMaskSh      	= fftshift(fftshift(fftshift(kMask,1),2),3);
dktMeas     	= F3_k2x(dktMeas);
dktMeas       	= F3_x2k(dktMeas,true);

%% iterative deconvolution

% forward operator
oper_K          = @(x)forward_model(x,sxtRef,kMaskSh,conv_sz_cell,Mask_cell,senseMap,lambda_ref);
oper_K_adj      = @(x)adjoint_model(x,sxtRef,kMaskSh,conv_sz_cell,Mask_cell,senseMap,lambda_ref);

% check adjoint
if disp
    rel_err     = checkAdjoint(oper_K,oper_K_adj,[sum(len),1],1e-4,1);
end

% construct the linear system
linear_sys      = @(x,mode)prep_Afun(x,mode,oper_K,oper_K_adj); %@@ use lsmr

% prepare data
d               = oper4D_mask_4D_data(dktMeas,kMaskSh);
d               = [d;sqrt(lambda_ref)*img_ref(:)];

% lsmr
x               = lsmr(linear_sys, d(:), 0, cgtol, cgtol, [], maxit, [], disp); %@@ use lsmr

%% synthesize recon signal

% GS-based spatial modulation
y             = 0;
Ind           = 1;
for nm = 1 : NofRegion
    x_tmp     = x(Ind:Ind+len(nm)-1);                       % extract
    x_tmp     = reshape(x_tmp,conv_sz_cell{nm});            % reshape
    x_tmp     = ktrunc(x_tmp,[Ny,Nx,Nz]);                   % zero-padding
    x_tmp     = F3_k2x(x_tmp);                         % Fourier-transform
    x_tmp     = applySpatialSupport3d(x_tmp,Mask_cell{nm}); % apply tissue mask
    y         = y+x_tmp;
    Ind       = Ind + len(nm);
end
y             = y * sqrt(Ny*Nx*Nz);                         % scale

% apply GS-modulation to reference signal
sxtRecon      = bsxfun(@times,sxtRef,y);

end

function y = forward_model(x,sxtRef,kMask,conv_sz_cell,Mask_cell,senseMap,lambda_ref)

%% data size info
[Ny,Nx,Nz,Nt] = size(kMask);
Nc            = size(senseMap,5);
NofRegion     = length(Mask_cell);
len           = zeros(NofRegion,1);
for nm = 1 : NofRegion
    len(nm)   = prod(conv_sz_cell{nm});
end

%% forward pass

% GS-based spatial modulation
y             = 0;
Ind           = 1;
for nm = 1 : NofRegion
    x_tmp     = x(Ind:Ind+len(nm)-1);                       % extract
    x_tmp     = reshape(x_tmp,conv_sz_cell{nm});            % reshape
    x_tmp     = ktrunc(x_tmp,[Ny,Nx,Nz]);                   % zero-padding
    x_tmp     = F3_k2x(x_tmp);                         % Fourier-transform
    x_tmp     = applySpatialSupport3d(x_tmp,Mask_cell{nm}); % apply tissue mask
    y         = y+x_tmp;
    Ind       = Ind + len(nm);
end
y             = y * sqrt(Ny*Nx*Nz);                         % scale

% apply GS-modulation to reference signal
y             = bsxfun(@times,sxtRef,y);
y_reg         = y(:,:,:,1);

% apply sensitivity map
y             = bsxfun(@times,senseMap,y);

% Fourier transform
y             = F3_x2k(y,true);

% sampling mask
y             = oper4D_mask_4D_data(y,kMask);

% vectorize
y             = y(:);

% regularization
y_reg         = sqrt(lambda_ref) * y_reg(:);

% combine output
y             = [y;y_reg];

end

function y = adjoint_model(x,sxtRef,kMask,conv_sz_cell,Mask_cell,senseMap,lambda_ref)

%% data size info
[Ny,Nx,Nz,Nt] = size(kMask);
Nc            = size(senseMap,5);
NofRegion     = length(Mask_cell);
len           = zeros(NofRegion,1);
for nm = 1 : NofRegion
    len(nm)   = prod(conv_sz_cell{nm});
end

%% decompose input vector
x_reg         = x(1+sum(kMask(:))*Nc:end);
x             = x(1:sum(kMask(:))*Nc);

%% adjoint pass

% zero-filling
y             = oper4D_mask_4D_data_adj(x,kMask);

% Fourier transform
y             = F3_k2x(y,true);

% apply sensitivity map
y             = sum(bsxfun(@times,conj(senseMap),y),5);

% regularization
y_reg         = reshape(sqrt(lambda_ref) * x_reg(:),[Ny,Nx,Nz]);
y_reg         = bsxfun(@times,conj(sxtRef(:,:,:,1)),y_reg);

% apply GS-modulation to reference signal
y             = sum(bsxfun(@times,conj(sxtRef),y),4);

% combine with regularization
y             = y + y_reg;

% GS-based spatial modulation
y             = y * sqrt(Ny*Nx*Nz);                         % scale
y1            = zeros(sum(len),1,'like',x);
Ind           = 1;
for nm = 1 : NofRegion
    y_tmp     = applySpatialSupport3d(y,Mask_cell{nm});     % apply tissue mask
    y_tmp     = F3_x2k(y_tmp);                         % Fourier-transform
    y_tmp     = ktrunc(y_tmp,conv_sz_cell{nm});             % zero-padding
    y1(Ind:Ind+len(nm)-1) = y_tmp(:);
    Ind       = Ind + len(nm);
end

% output
y             = y1;


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

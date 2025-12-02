function [t1map,pdmap,sxtT1] = estT1map3DfromFlashVFA(imgFlh,FAflh,TRflh,mask,opt)
%% =======================================================================
% Function to estimate T1 map from FLASH data using VFA method
% Input: 
%   imgFlh: FLASH images with different flip angle:     [Ny,Nx,Nz,FA]
%   FAflh:  Array of flip angle (should match data):    [FA,1], Unit: Deg
%   TRflh:  TR of the FLASH sequence:                   [1], Unit: s
%   mask:   spatial support for fitting:                [Ny,Nx,Nz]
%   opt:    options (for extension)
% Output: 
%   t1map:  T1 map
%   pdmap:  PD map
% Note: 
%   The basic equition for the relationship between FLASH signal and flip
%   angle is: 
%       S(a) = (M0*(1-E1)*sin(a))/(1-E1*cos(a)) 
%   where E1 = exp(-TR/T1)
% 2020-11-11:
%   Add fitting with multiple frames
% 2020-12-03:
%   Add fitting with multiple frames (fixed)
%                                                         R.Guo @ UIUC
% ------------------------------------------------------------------------

%% basic check 
% size of data
[Ny,Nx,Nz,Nfa,Nfm] = size(imgFlh);
if(Nfa<2)
    error('number of flip angle should be larger than 1 !/n');
end
% size of flip angle
if(length(FAflh)~=Nfa)
    error('input flip angle number should match data !/n');
end
FAflh = vec(FAflh/180*pi);
% default mask
if((nargin<4)||(isempty(mask)))
    mask = ones(Ny,Nx,Nz)>0;
end
if((size(mask,1)~=Ny)||(size(mask,2)~=Nx)||(size(mask,3)~=Nz))
    mask = imresize3d(mask,[Ny,Nx,Nz]);
end
% default option
if(nargin<5)
    opt = struct;
end
verbose     = true;
realValue   = true;

structKey   = {'verbose','realValue'};
structVal   = { verbose , realValue };
opt = SetStructDefault(opt,structKey,structVal);

verbose     = opt.verbose;
realValue   = opt.realValue;

if(Nfm>1)
    FAflh = repmat(FAflh,[Nfm,1]);
    imgFlh = reshape(imgFlh,[Ny,Nx,Nz,Nfa*Nfm]);
end

%% T1 fitting
% variable setup
datVecFlh   = maskApplyNDdata3Dmask(imgFlh,mask);
NX          = size(datVecFlh,1);
T1Vec       = zeros(NX,1,'like',datVecFlh);
PDVec       = zeros(NX,Nfm,'like',datVecFlh);
% ones matrix setup 
onesMat     = ones(Nfa,1);
if(Nfm>1)
    onesMat = repmat(eye(Nfm),[1,1,Nfa]);
    onesMat = permute(onesMat,[3,2,1]);
    onesMat = reshape(onesMat,Nfa*Nfm,Nfm);
end
% point by point fitting
parfor iX = 1:NX
    % linear fitting
    s_t         = vec(datVecFlh(iX,:));
    b           = s_t./sin(FAflh);
    x           = s_t./tan(FAflh);
    A           = [x,onesMat];
    Ainv        = pinv(A);
    c           = Ainv*b;
    T1Vec(iX)   = -TRflh/log(c(1));
    PDVec(iX,:) = c(2:end)/(1-exp(-TRflh/T1Vec(iX)));
end
% back to 3D map 
t1map   = maskRemoveNDdata3Dmask(T1Vec,mask);
pdmap   = maskRemoveNDdata3Dmask(PDVec,mask);
% real value of T1
if(realValue)
    t1map   = real(t1map);
end
if(nargout>2)
    FAflh2 = reshape(FAflh(1:Nfa),1,1,1,[]);
    pdmap2 = reshape(pdmap,Ny,Nx,Nz,1,Nfm);
    E1 = exp(-TRflh/(t1map+eps));
    sxtT1 = bsxfun(@times,(1-E1),sin(FAflh2))./(1-bsxfun(@times,E1,cos(FAflh2)));
    sxtT1 = bsxfun(@times,pdmap2,sxtT1);
    if(verbose>1)
        sxtT1h = reshape(sxtT1,Ny,Nx,Nz,Nfa*Nfm);
        cmp_voxel_spectrum3d(imgFlh,sxtT1h);
    end
end


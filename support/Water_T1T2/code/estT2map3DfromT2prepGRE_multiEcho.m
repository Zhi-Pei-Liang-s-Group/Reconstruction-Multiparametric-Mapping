function [t2map,pdmap] = estT2map3DfromT2prepGRE_multiEcho(sxtPrep,t1map,pdmap,T2prep,FA,TR,mask,opt)
%% =======================================================================
% Function to estimate T2 map from T2prep-GRE data
% Input: 
%   sxtPrep: T2-prep GRE images with different T2prep:  [Ny,Nx,Nz,Nt]
%   t1map:  pre-determined T1 map:                      [Ny,Nx,Nz]
%   T2prep: vector of T2-prep time:                     [Nt,1], Unit: sec
%   FA:     flip angle:                                 [1], Unit: Deg
%   TR:     repeat time:                                [1], Unit: s
%   mask:   spatial support for fitting:                [Ny,Nx,Nz]
%   opt:    options (for extension)
% Output: 
%   t2map:  T2 map
%   pdmap:  PD map (not true PD)
% Note: 
%   The basic equition for the T2prep signal is: 
%       S(t) = (M0*(1-E1)*E2*sin(a))/(1-E1*E2*cos(a)) 
%   where E1 = exp(-TR/T1); E2 = exp(-T2prep/T2)
%                                                         R.Guo @ UIUC
% ------------------------------------------------------------------------

%% basic check 
% size of data
[Ny,Nx,Nz,Nf,M] = size(sxtPrep); % M is the number of echoes
if(Nf<2)
    error('number of T2-prep should be larger than 1 !/n');
end
% size of flip angle
if(length(T2prep)~=Nf)
    error('input T2-prep number should match data !/n');
end
T2prep = vec(T2prep);

% default mask
if((nargin<6)||(isempty(mask)))
    mask = ones(Ny,Nx,Nz)>0;
end
if((size(mask,1)~=Ny)||(size(mask,2)~=Nx)||(size(mask,3)~=Nz))
    mask = imresize3d(mask,[Ny,Nx,Nz]);
end
% T1 map 
if((size(t1map,1)~=Ny)||(size(t1map,2)~=Nx)||(size(t1map,3)~=Nz))
    t1map = imresize3d(t1map,[Ny,Nx,Nz]);
end
% PD map 
if((size(pdmap,1)~=Ny)||(size(pdmap,2)~=Nx)||(size(pdmap,3)~=Nz))
    pdmap = imresize3d(pdmap,[Ny,Nx,Nz]);
end
% default option
if(nargin<8)
    opt = struct;
end
verbose     = true;
absValue    = true;

structKey   = {'verbose','absValue'};
structVal   = { verbose , absValue };
opt = SetStructDefault(opt,structKey,structVal);

verbose     = opt.verbose;
absValue    = opt.absValue;

%% T1 fitting
sxtPrep     = abs(sxtPrep);
fa          = FA/180*pi;
tr          = TR;
tp          = T2prep; 
% variable setup
datVec      = reshape(abs(maskApplyNDdata3Dmask(sxtPrep,mask)),[],Nf,M);
t1Vec       = maskApplyNDdata3Dmask(t1map,mask);
pdVec       = maskApplyNDdata3Dmask(pdmap,mask);
NX          = size(datVec,1);
t2Vec       = zeros(NX,1,'like',datVec);
% point by point fitting
parfor iX = 1:NX
    % linear fitting
    b1 = [];
    for it = 1:M
        s1      = vec(datVec(iX,:,it));
        t1      = t1Vec(iX,:);
        expt1   = exp(-tr./t1);
        c1      = (1-expt1)*sin(fa);
        c2      = expt1*cos(fa);
        a0      = pdVec(iX,:);
        b1      = [b1;log(s1./(c1*a0+c2*s1))];
    end
    tp_tmp  = vec(repmat(tp,[1,M]));
    r1      = -pinv(tp_tmp)*b1;
    t2      = 1./(r1);
    t2Vec(iX) = t2;
end
% back to 3D map 
t2map   = maskRemoveNDdata3Dmask(t2Vec,mask);
pdmap   = maskRemoveNDdata3Dmask(pdVec,mask);
if(absValue)
    t2map   = abs(t2map);
end


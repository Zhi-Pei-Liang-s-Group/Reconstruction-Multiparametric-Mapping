function sxt_syn = synFlashVFAfromT1PDMap(t1map,pdmap,FAflh,TRflh,mask,opt)
% This script synthesizes VFA images based on T1 and pdmap

%% basic check 

% size of flip angle
Nfa                = length(FAflh);
FAflh = vec(FAflh/180*pi);

% size of data
[Ny,Nx,Nz]         = size(t1map);

% default mask
if((nargin<4)||(isempty(mask)))
    mask = ones(Ny,Nx,Nz)>0;
end
if((size(mask,1)~=Ny)||(size(mask,2)~=Nx)||(size(mask,3)~=Nz))
    mask = imresize3d(mask,[Ny,Nx,Nz]);
end

% default option
if(nargin<6)
    opt = struct;
end
verbose     = true;
realValue   = true;

structKey   = {'verbose','realValue'};
structVal   = { verbose , realValue };
opt = SetStructDefault(opt,structKey,structVal);

verbose     = opt.verbose;
realValue   = opt.realValue;

%% VFA image synthesis

% variable setup
t1map       = applySpatialSupport3d(t1map,mask);
pdmap       = applySpatialSupport3d(pdmap,mask);

% synthesis
FAflh2      = reshape(FAflh(1:Nfa),1,1,1,[]);
pdmap2      = reshape(pdmap,Ny,Nx,Nz);
E1          = exp(-TRflh/(t1map+eps));
sxt_syn     = bsxfun(@times,(1-E1),sin(FAflh2))./(1-bsxfun(@times,E1,cos(FAflh2)));
sxt_syn     = bsxfun(@times,pdmap2,sxt_syn);   



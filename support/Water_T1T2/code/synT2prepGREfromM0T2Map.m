function sxt_syn = synT2prepGREfromM0T2Map(t2map,pdmap,t1map,T2prep,FA,TR,mask,opt)
% This script synthesizes T2prep mGRE images based on T1 and pdmap

%% basic check 
% size of data
[Ny,Nx,Nz] = size(t2map); % M is the number of echoes

% size of T2 preparation
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

%% image synthesis
fa          = FA/180*pi;
tr          = TR;
tp          = T2prep.'; 

% signal synthesis
E1          = exp(-tr./t1map(:));
E2          = exp(-(1./t2map(:)) * tp);
sxt_syn     = bsxfun(@times,pdmap(:) .* (1-E1) * sin(fa), E2) ./ (1 - bsxfun(@times, E2, E1 * cos(fa)));
sxt_syn     = reshape(sxt_syn,Ny,Nx,Nz,[]);


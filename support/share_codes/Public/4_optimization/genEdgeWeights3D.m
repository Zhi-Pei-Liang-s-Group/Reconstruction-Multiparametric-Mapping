function [edgeMap,edgeMat] = genEdgeWeights3D(imag3D,edgeMode,mask3D,opt)
%% ========================================================================
% [edgeMap,edgeMat] = genEdgeWeights3D(imag3D,edgeMode,mask3D,opt)
% 
% Generate edge weights from 3D images
%
% Input: 
%   imag3D:     anatomical images           size: [Ny,Nx,Nz]
%   edgeMode:   mode for generate edgemap 
%       0:  first order gradient map
%       1:  canny edge only 
%       2:  grdient map + canny edge 
%       3:  second order gradient map 
%       4:  to be added
%   mask3D: spatial mask                    size: [Ny,Nx,Nz,Nt2]
%   opt: 
%       ratioGrad:  ratio to gen gradient in 0 and 2
%       ratioComb2: ratio for gradient field in mode 2
%       verbose:    flag to display
% Output: 
%   edgeMap: output edge maps               size: [Ny,Nx,Nz,dirt]
%   edgeMat: output edge map matrix    
%
% Note: 
%   In the output edgemap, 0 is edge, 1 is empty region
%   
%                                                   R.GUO @ UIUC
% ------------------------------------------------------------------------- 

% basic setup 
[Ny,Nx,Nz,~]  = size(imag3D);

if((~exist('edgeMode','var'))||(isempty(edgeMode)))
    edgeMode = 0;
end
if((~exist('mask3D','var'))||(isempty(mask3D)))
    mask3D  = ones(Ny,Nx,Nz)>0;
end
if(~(all(size_dims(mask3D,1:3)==[Ny,Nx,Nz]))) % @@ Yibo: for 2D and 1D cases
    mask3D = imresize3d(mask3D,[Ny,Nx,Nz])>0;
end
if(~isa(imag3D,'double'))
    imag3D = double(imag3D);
end
if(~isreal(imag3D))
    imag3D = abs(imag3D);
end

% default option
ratioGrad   = 30;
ratioComb2  = 0.7;
verbose     = false;

if(exist('opt','var')&&(~isempty(opt))) 
    structKey   = {'ratioGrad','ratioComb2','verbose'};
    structVal   = { ratioGrad , ratioComb2 , verbose };
    opt         = SetStructDefault(opt,structKey,structVal);
    ratioGrad   = opt.ratioGrad;
    verbose     = opt.verbose;
    ratioComb2  = opt.ratioComb2;
end

% detect image dimension @@ Yibo: for 2D and 1D cases
if isvector(imag3D) 
    Ndim        = 1;
elseif ismatrix(imag3D)
    Ndim        = 2;
elseif ndims(imag3D)==3
    Ndim        = 3;
else
    error('Unsupported data size.');
end

% generating edge weights based on the modes
if(edgeMode == 0)
% mode 0, first order gradient map
    edgeMap     = genGradientMap3D(imag3D,ratioGrad,mask3D);
elseif(edgeMode == 1)
% mode 1, canny 3D edges only
    assert(Ndim>1,'Cannot process 1D reference image in edge mode 1.');
    edgeMap     = 1 - canny(imag3D.*mask3D,[],[]);
    edgeMap     = repmat(edgeMap,[1,1,1,Ndim]);
elseif(edgeMode == 2)
% mode 2, grdient map + canny edge 
    assert(Ndim>1,'Cannot process 1D reference image in edge mode 2.');
    edgeMap1    = genGradientMap3D(imag3D,ratioGrad,mask3D);
    edgeMap2    = repmat(canny(imag3D.*mask3D,[],[]),[1,1,1,Ndim]);
    edgeMap     = (1-ratioComb2) + ratioComb2*edgeMap1 - edgeMap2;
elseif(edgeMode == 3) 
% mode 3, second order gradient map
    edgeMap1    = genGradientMap3D(imag3D,ratioGrad,mask3D);
    edgeMap     = repmat(edgeMap1,[1,1,1,1,Ndim]); % @@ Yibo: for 2D and 1D cases
    for idir = 1:Ndim
        edgeMap(:,:,:,:,idir) = genGradientMap3D(edgeMap1(:,:,:,idir),ratioGrad,mask3D);
    end
    edgeMap     = normRange(reshape(edgeMap,Ny,Nx,Nz,[]));
else
    error('Not supported edge mode\n');
end

% display
if(verbose)
    figure,montagesc(sumOfSqr(edgeMap,4));
    colormap gray,axis image off;
    title('aligned composite edge weightes')
end

% generate weight diff matrix 
if(nargout>1)
    assert(edgeMode ~= 3,'Cannot generate edgeMat in edge mode 3.');
    D           = Diffmat_PeriodicBoundary(Ny,Nx,Nz);
    edgeMat     = sparsediag(sqrt(vec(edgeMap)))*D;
end

end



%% ========================================================================
% Support functions
% -------------------------------------------------------------------------

function edgeWeight = genGradientMap3D(imag3D,ratioGrad,mask3D)
% -------------------------------------------------------------------------
% Generate first order gradient map from 3D image
% -------------------------------------------------------------------------

% size setup
[Ny,Nx,Nz,~]    = size(imag3D);
Nyxz            = Ny*Nx*Nz;
if isvector(imag3D) % @@ Yibo: for 2D and 1D cases
    Ndir        = 1;
elseif ismatrix(imag3D)
    Ndir        = 2;
elseif ndims(imag3D)==3
    Ndir        = 3;
else
    error('Unsupported data size.');
end
% diff matrix setup
[D,~]           = Diffmat_PeriodicBoundary(Ny,Nx,Nz);
% extract the edges
edgeWeight      = zeros(Ny,Nx,Nz,Ndir,'like',imag3D);    
edges           = abs(D*reshape(imag3D,[],1));   
% adjust to ratio
alpha           = max(edges(:))/ratioGrad;
w               = 1.*(edges < alpha) + (alpha./edges).*(edges>=alpha); 
w(isnan(w))     = 1;
w(isinf(w))     = 1;                               
% apply mask and pre-process
for ir  = 1:Ndir
    wi          = reshape(w((ir-1)*Nyxz+1:ir*Nyxz),Ny,Nx,Nz);
    wi          = wi.*mask3D;   
    wi(wi==0)   = max(wi(:));      
    wi(isnan(wi))  = max(wi(:));
    wi(wi<=0)      = max(wi(:));     
    edgeWeight(:,:,:,ir) = wi;
end 
% normalize the edge map
edgeWeight      = edgeWeight./max(edgeWeight(:));
end


function [D,Dp] = Diffmat_PeriodicBoundary(N1,N2,N3)
% Generates the sparse three-dimensional finite difference (first-order neighborhoods) matrix D 
% for an image of dimensions N1xN2xN3 (rows x columns x slices).  The optional output
% output argument Dp is the transpose of D.  Also works for 2D images if N3 is 1.
% ------ Fan Lam 15/08/2012 ------ %
% Change-log
% (2015-09-22, Bryan Clifford) Swap out the missing "GetDiffMat" with Diffmat_periodicBoundary2D
%-------------------------------------------------------------------------------

if (not(isreal(N1)&&(N1>0)&&not(N1-floor(N1))&&isreal(N2)&&(N2>0)&&not(N2-floor(N2))))
    error('Inputs must be real positive integers');
end
if ((N1==1)&&(N2==1)&&(N3==1))
    error('Finite difference matrix can''t be generated for a single-pixel image');
end

D1 = [];
D2 = [];
D3 = [];

if (N1 > 1)&&(N2>1)&&(N3>1)    
    e = ones(N1,1);
    if (numel(e)>2)
        T = spdiags([e,-e],[0,1],N1,N1);
        T(N1,1)=-1;
        E = speye(N2);
        E2 = speye(N3);
        D1 = kron(E2,kron(E,T)); % column wise finite difference
    end
    e = ones(N2,1);
    if (numel(e)>2)
        T = spdiags([e,-e],[0,1],N2,N2);
        T(N2,1)=-1;
        E = speye(N1);
        E2 = speye(N3);
        D2 =  kron(E2,kron(T,E)); % row wise finite difference
    end
    e = ones(N3,1);
    if (numel(e)>2)
        T = spdiags([e,-e],[0,1],N3,N3); % slice wise finite difference
        T(N3,1)=-1;
        E = speye(N1);
        E2 = speye(N2);
        D3 =  kron(T,kron(E2,E));
    end
    D = [D1;D2;D3];
elseif (N1 > 1)&&(N2>1)&&(N3 == 1)    % for 2D Image
    [Dv, Dh] = Diffmat_PeriodicBoundary2D(N1,N2,1);
    D = [Dv; Dh];
else
    D = Diffmat_PeriodicBoundary1D_local(N1,1);
%     error('singleton dimensions not supported');
end

if (nargout > 1)
    Dp = D';
end
end


function [Dv,Dh] = Diffmat_PeriodicBoundary2D(N1,N2,flag)
% flag = 1: enable periodic boundary; 0: disable periodic boundary
if flag == 1
    e = ones(N1,1);
    T = spdiags([e,-e],[0,1],N1,N1);
    T(N1,1)=-1;
    E = speye(N2);
    Dv = kron(E,T);%column wise finite difference
    
    clear T
    clear E
    
    e = ones(N2,1);
    T = spdiags([e,-e],[0,1],N2,N2);
    T(N2,1)=-1;
    E = speye(N1);
    Dh =  kron(T,E);%row wise finite difference
else
    e = ones(N1,1);
    T = spdiags([e,-e],[0,1],N1,N1);
    E = speye(N2);
    Dv = kron(E,T);%column wise finite difference

    T = spdiags([e,-e],[0,1],N2,N2);
    E = speye(N1);
    Dh =  kron(T,E);%row wise finite difference
end
end

function D = Diffmat_PeriodicBoundary1D_local(N,flag) % @@ Yibo: for 1D cases
% flag = 1: enable periodic boundary; 0: disable periodic boundary
    if nargin<2
        flag = 1;
    end

    % generate derivative operator
    e = ones(N,1);
    D = full(spdiags([e,-e],[0,1],N,N));
    switch flag
        case 0
            D = D(1:end-1,:);
        case 1
            D(N,1)=-1;
    end
end
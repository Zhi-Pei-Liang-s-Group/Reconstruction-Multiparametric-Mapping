function [Dg,edgeMap] = genWL2mtx(anatRef,opt)
%% ==================================================================
%GENWL2MTX generate a matrix for weighted-L2 regularization
% This function is now a repackage of genEdgeWeights3D, which you may directly use.
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-07-22
%
%   [INPUTS]
%   ---- Required ----
%   anatRef                 anatomical reference [y,x,z]
%
%   ---- Optional ----
%   opt:
%     - mask                mask inside which to generate the matrix
%     - edgeMode            mode to estimate the edges [0]
%     - ratio               ratio of edges [30]
%     - ratio2              ratio of edges in mode 2 [0.7]
%     - verbose             degree of verbose [0]
%
%   [OUTPUTS]
%   Dg                      regularization matrix
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/07/22
%       Modified by Yibo Zhao @ UIUC, 2019/02/05:
%           Change the order of if statements to process 1D data
%       Modified by Yibo Zhao @ UIUC, 2019/07/01:
%           Use genEdgeWeights3D from Rong Guo @ UIUC for edge estimation
%           to enable other modes, see genEdgeWeights3D.m for more details
%       Modified by Yudu Li @ UIUC, 2020/11/26:
%           Output edgemap
%
%   See also genEdgeWeights3D
%
%   Example: (suggested edgeMode: 0 or 2)
%       Dg = genWL2mtx(anatRef,struct('ratio',30,'verbose',1,'mask',brainMask,'edgeMode',0));
%       Dg = genWL2mtx(anatRef,struct('ratio',30,'ratio2',0.7,'verbose',1,'mask',brainMask,'edgeMode',2));
%
%--------------------------------------------------------------------------

%% ------ parse the input ------

    dims = size(anatRef);
    
    if ~exist('opt','var')||isempty(opt)
        opt = struct;
    end
    opt = SetStructDefault(opt,...
        {    'mask','ratio','ratio2','verbose','edgeMode'},...
        {true(dims),     30,     0.7,        0,         0});
    
    mask       = opt.mask;
    ratio      = opt.ratio;
    ratioComb2 = opt.ratio2;
    verbose    = opt.verbose;
    edgeMode   = opt.edgeMode;
    
    if ~(all(size(mask)==dims))
        mask = imresize3d(mask,dims)>0;
    end
    
    if ~isa(anatRef,'double')
        anatRef = double(anatRef);
    end
    
    if ~isreal(anatRef)
        anatRef = abs(anatRef);
    end
    
%% --- generate Dg ---
    opt_edge = struct('ratioGrad',ratio,'ratioComb2',ratioComb2,'verbose',verbose);
    [edgeMap,Dg]   = genEdgeWeights3D(anatRef,edgeMode,mask,opt_edge);
    
end

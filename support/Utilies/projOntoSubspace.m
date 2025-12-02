function [sxt_proj,cx_proj] = projOntoSubspace(sxt,V_subspace,dim_t_data,dim_t_V,lambda)
%% ======================================================================
%PROJONTOSUBSPACE project a (spatio-)temporal domain data onto a subspace
% =======================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-10-25
%
%   [INPUTS]
%   ---- Required ----
%   sxt                     (x-)t domain data
%   V_subspace              basis
%
%   ---- Optional ----
%   dim_t_data              the dimension of t in data [4]
%   dim_t_basis             the dimension of t in basis [2]
%
%   [OUTPUTS]
%   sxt_proj                projected data
%   cx_proj                 fitted projection coefficients
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/10/25
%       Modified by Yibo Zhao @ UIUC, 2019/01/31:
%           Add a new output, cx_proj
%       Modified by   Yudu Li @ UIUC, 2019/09/13: 
%           Add L2 regularization
%       Modified by Yibo Zhao @ UIUC, 2019/10/21:
%           Change the code to the convention of pre-multiplication on column vectors
%       Modified by Yibo Zhao @ UIUC, 2021/03/16:
%           Update for uoss
%
%   See also SUBSPACEANALYZE, SUBSPACESYNTHESIZE
%
%--------------------------------------------------------------------------

    
%% ------ parse the input ------
    if ~exist('dim_t_data','var')||isempty(dim_t_data)
        dim_t_data = 4;
    end

    if ~exist('dim_t_V','var')||isempty(dim_t_V)
        dim_t_V = 2;
    end
    if ~exist('lambda','var')||isempty(lambda)
        lambda  = 0;
    end
    
%% ------ handle the case of uoss ------
    if iscell(V_subspace)
        if nargout==1
            sxt_proj = projOntoSubspace(sxt,cell2mat(V_subspace),dim_t_data,dim_t_V,lambda);
        else
            [sxt_proj,cx_mat] = projOntoSubspace(sxt,cell2mat(V_subspace),dim_t_data,dim_t_V,lambda);
            N_comp    = length(V_subspace);
            Rank_comp = zeros([N_comp,1]);
            for ii = 1:N_comp
                Rank_comp(ii) = size(V_subspace{ii},3-dim_t_V);
            end
            inds_comp = length2ind(Rank_comp);
            cx_proj = cell(1,N_comp);
            for ii = 1:N_comp
                cx_proj{ii} = getDataIndices(cx_mat,dim_t_data,inds_comp{ii});
            end
        end
        return;
    end
    
%% --- project onto the subspace ---
    assert(size(sxt,dim_t_data)==size(V_subspace,dim_t_V),'Data and basis temporal lengths must be the same.');
    
% [in the convention of pre-multiplication on COLUMN vectors]

    % permute time to the last dimension
    perm_order             = 1:ndims(sxt);
    perm_order(end)        = dim_t_data;
    perm_order(dim_t_data) = ndims(sxt);
    sxt_perm               = permute(sxt,perm_order); % [y,x,z,coil,avg,...,t]

    % take care of the basis
    switch dim_t_V
        case 1
            Vtr = V_subspace;   % [txr]
        case 2
            Vtr = V_subspace.'; % [txr]
        otherwise
            error('The fourth input must be either 1 or 2.');
    end
    
    if lambda == 0
        
        sxt_proj = reshape(sxt_perm,[],size(sxt,dim_t_data)).'; % [txP]
%         cx_proj  = pinv(Vtr)*sxt_proj;                          % [rxt] * [txP] = [rxP]
        cx_proj  = Vtr\sxt_proj;                          % [rxt] * [txP] = [rxP]
        sxt_proj = Vtr*cx_proj;                                 % [txr] * [rxP] = [txP]
    else
        
        sxt_proj  = reshape(sxt_perm,[],size(sxt,dim_t_data)).';
        linear_A  = cat(1,Vtr,sqrt(lambda)*eye(size(Vtr,2)));
        linear_b  = cat(1,sxt_proj,zeros(size(Vtr,2),size(sxt_proj,2)));
        cx_proj   = pinv(linear_A)*linear_b;
        sxt_proj  = Vtr*cx_proj;
        
    end
    
    if nargout>1
        % assume 1:dim_t_data-1 are all spatial dimensions
        if dim_t_data>1
            cx_proj  = reshape(cx_proj.',[size_dims(sxt_perm,1:dim_t_data-1),size(Vtr,2)]);
        end
    end
    sxt_proj = reshape(sxt_proj.',size(sxt_perm));          % [y,x,z,coil,avg,...,t]
    sxt_proj = permute(sxt_proj,perm_order);
    
end
    
% [in the convention of post-multiplication on ROW vectors]
%{
    % permute time to the last dimension
    perm_order             = 1:ndims(sxt);
    perm_order(end)        = dim_t_data;
    perm_order(dim_t_data) = ndims(sxt);
    sxt_perm               = permute(sxt,perm_order); % [y,x,z,coil,avg,...,t]

    % take care of the basis
    switch dim_t_V
        case 1
            Vrt = V_subspace.'; % [rxt]
        case 2
            Vrt = V_subspace;   % [rxt]
        otherwise
            error('The fourth input must be either 1 or 2.');
    end
        
    sxt_proj = reshape(sxt_perm,[],size(sxt,dim_t_data)); % [Pxt]
    cx_proj  = sxt_proj*pinv(Vrt);                        % [Pxt] * [txr] = [Pxr]
    sxt_proj = cx_proj*Vrt;                               % [Pxr] * [rxt] = [Pxt]

    if nargout>1
        % assume 1:dim_t_data-1 are all spatial dimensions
        if dim_t_data>1
            cx_proj  = reshape(cx_proj,[size_dims(sxt_perm,1:dim_t_data-1),size(Vrt,1)]);
        end
    end
    sxt_proj = reshape(sxt_proj,size(sxt_perm));          % [y,x,z,coil,avg,...,t]
    sxt_proj = permute(sxt_proj,perm_order);
%}
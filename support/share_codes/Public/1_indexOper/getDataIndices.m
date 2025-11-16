function data = getDataIndices(data,dim,indices)
%% ========================================================================
%GETDATAINDICES get the values of data in selected indices along desired
%dimensions
%Note: this function is inefficient, so avoid using it in a loop.
% =========================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-07-22
%
%   [INPUTS]
%   ---- Required ----
%   data                    data to select
%   dim                     dimension along which to select indices
%
%   ---- Optional ----
%   indices                 indices to select [cenInd(size(data,dim))]
%
%   [OUTPUTS]
%   dataSel                 selected data
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/07/22
%       Modified by Yibo Zhao @ UIUC, 2021/07/14:
%           Allow accessing extra dimensions if the index is 1.
%
%   Example:
%       Say, we have a 5D data sxtc, and [Ny,Nx,Nz,Nt,Nc] = size(sxtc).
%       1. getDataIndices(sxtc,4,1) is equivalent to sxtc(:,:,:,1,:)
%       2. getDataIndices(sxtc,4,1:2:Nt) is equivalent to sxtc(:,:,:,1:2:Nt,:)
%       3. getDataIndices(sxtc,[3,4],{12,1}) is equivalent to sxtc(:,:,12,1,:)
%       4. getDataIndices(sxtc,[1,2,3],{cenInd(Ny),cenInd(Nx),cenInd(Nz)}) 
%          is equivalent to sxtc(cenInd(Ny),cenInd(Nx),cenInd(Nz),:,:)
%          ( in this case, you can simply use getDataIndices(sxtc,[1,2,3]))
%       This function is extremely useful when you don't want hard-code the 
%       dimensions to select.
%       Also, you can get the k-space center FID of a 4D x-t data by 
%       getDataIndices(F3_x2k(sxt),[1,2,3])), instead of using an 
%       intermediate variable.
%
%       Enjoy!
%
%   See also SETDATAINDICES, SUBSREF
%--------------------------------------------------------------------------
 
%% --- parse inputs ---
    try
        assert(all(dim<=ndims(data))||indices==1);
    catch
        error('Selected dimension exceeded the total number of dimensions.');
    end
    
    if ~exist('indices','var')||isempty(indices)
        if isscalar(dim)
            indices = cenInd(size(data,dim));
        elseif isvector(dim)
            for n=1:length(dim)
                indices{n} = cenInd(size(data,dim(n)));
            end
        else
            error('Input dim must be a scalar or vector.');            
        end
    end
    
    if iscell(indices)
        try
            assert(length(indices)==length(dim))
        catch
            error('When get multiple dimensions of data, length of dim must be the same as length of indices.');
        end
    end
    
%% --- select data ---
    if iscell(indices)
        for n = 1:length(indices)
            cmd_select = genCmdGetIndices('data',ndims(data),dim(n),'indices{n}','data');
            eval(cmd_select);
        end
    else
        cmd_select = genCmdGetIndices('data',ndims(data),dim,'indices','data');
        eval(cmd_select);
    end
    
end


%#ok<*STOUT>
%#ok<*NASGU>
%Future:
%  Build-in options: get odd, get even, get first half, get last half, get center...




function cmd = genCmdGetIndices(dataName,numDims,dim,indSelName,newDataName)
%% ========================================================================
%genCmdGetIndices generate a command to select indices along desired
%diemsnion
% =========================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-07-19
%
%   [INPUTS]
%   ---- Required ----
%   dataName                name of the data
%   numDims                 total number of dimensions of the data
%   dim                     dimension along which to select indices
%   indSelName              name of indices to select
%
%   ---- Optional ----
%   newDataName             name of new data [dataName]
%
%   [OUTPUTS]
%   cmd                     a command to select the indices
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/07/19
%       Modified by Yibo Zhao @ UIUC, 2021/07/14:
%           Allow accessing extra dimensions if the index is 1.
%           
%   Example - If I want to select center 20 points of ktData along second dimension:
%       ind_cen20  = cenInd(size(ktData,2),20);
%       cmd_select = genCmdGetIndices(getName(ktData),ndims(ktData),2,getName(ind_cen20) );
%       eval(cmd_select);
%
%   This function seems clumsy since you can simply run ktData = ktData(:,ind_cen20,:);
%   But it could be very useful if you want to write your code more generally ;)
%--------------------------------------------------------------------------

%% --- parse inputs ---
    if ~exist('newDataName','var')||isempty(newDataName)
        newDataName = dataName;
    end
    
    try
        assert(all(dim<=numDims))
    catch
        warning('Selected dimension exceeded the total number of dimensions. This is only possible when the index is 1.');
    end
    
%% --- generate the command ---

    cmd = sprintf('%s = %s(',newDataName,dataName);
    for n = 1:numDims
        if n ~= numDims % not the last dimension
            if n ~= dim
                cmd = sprintf('%s:,',cmd);
            else
                cmd = sprintf('%s%s,',cmd,indSelName);
            end
        else % the last dimension
            if n ~= dim
                cmd = sprintf('%s:',cmd);
            else
                cmd = sprintf('%s%s',cmd,indSelName);
            end
        end
    end
    cmd = sprintf('%s);',cmd);
        
end
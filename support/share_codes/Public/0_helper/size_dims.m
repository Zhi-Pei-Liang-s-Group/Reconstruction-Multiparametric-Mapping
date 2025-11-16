function varargout = size_dims(data,dims) 
%% ==================================================================
%SIZE_DIMS return size of data in selected dimensions
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-08-29
%
%   [INPUTS]
%   data                    data
%   dims                    selected dimensions
%
%
%   [OUTPUTS]
%   varargout               data size
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/08/29
%       Modified by Yibo Zhao @ UIUC, 2019/10/14
%           when a dimension is 0, return 1.
%           
%   Examples:
%   a) [L3,L1] = SIZE_DIMS(X,[3,1]) returns sizes of the third and first
%      dimensions of the array X. In this case, the number of outputs must be
%      the same as the length of dims.
%   b) dims_data = SIZE_DIMS(X,[1,2,3]) returns a 1-by-*3* vector containing the 
%      sizes of the array X along the first, second and thrid dimensions.
%      NOTE that size(X(:,:,:,1)) will return a 1-by-*2* vector when
%      size(X,3) = 1. This feature is very important, especially when you
%      want to write a code working for both 2D and 3D data.
%
%   See also: INDEXING_1D
%
%--------------------------------------------------------------------------

%%
    if nargout>1
        % when we have multiple outputs, we need to make sure the numbers match
        assert(length(dims)==nargout,'Mismatch of numbers of outputs and selected dimensions.');
        for n = 1:length(dims)
            if dims(n)~=0
                varargout{n} = size(data,dims(n));
            else
                varargout{n} = 1;
            end
        end
    else
        % otherwise just return an array of size
        varargout{1} = [];
        for n = 1:length(dims)
            if dims(n)~=0
                varargout{1} = [varargout{1},size(data,dims(n))];
            else
                varargout{1} = [varargout{1},1];
            end
        end
    end   
    
end

%#ok<*AGROW>

function sx_masked = applySpatialSupport3d(sx,mask)
%% ==================================================================
%APPLYSPATIALSUPPORT3D apply 3d spatial support to an image
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-07-19
%
%   [INPUTS]
%   ---- Required ----
%   sx                      image domain data [y,x,z,...]
%   mask                    spatial support [y,x,z]
%
%
%   [OUTPUTS]
%   sx_masked               masked image domain data [y,x,z,...]
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/07/19
%       Modified by Yibo Zhao @ UIUC, 2019/07/13:
%           When the mask is empty, don't do anything
%
%--------------------------------------------------------------------------
    
    if isempty(mask)
        sx_masked = sx;
        return;
    end
    
    dims3d    = size_dims(sx,[1,2,3]);
    if any(size_dims(mask,1:3) ~= dims3d)
        mask  = imresize3d(mask,dims3d);
    end
    sx_masked = bsxfun(@times,sx,mask);
    
end


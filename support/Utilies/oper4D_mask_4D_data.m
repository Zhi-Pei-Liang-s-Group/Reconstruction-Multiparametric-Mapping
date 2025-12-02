function y = oper4D_mask_4D_data(x,Mask)
%   Author: Yudu Li
%   Created: 03/15/2019
%   Decription: apply a 4D mask to a 4D matrix x; then change the resulting
%   values into a col vector.
%   Interface:
%           y = oper3D_mask_4D_data(x,Mask)
%   Input: 
%           x--L1xL2xL3xN, numerical
%           Mask--L1xL2xL3, boolean
%   Output:
%           y--column vector

[L1, L2, L3, N, Nc] = size(x);
assert(size(Mask,1)==L1&&size(Mask,2)==L2&&size(Mask,3)==L3&&size(Mask,4)==N);
Mask        = repmat(Mask,[1,1,1,1,Nc]);
y           = x(Mask(:));
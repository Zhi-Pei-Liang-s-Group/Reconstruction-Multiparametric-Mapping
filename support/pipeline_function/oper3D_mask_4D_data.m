function y = oper3D_mask_4D_data(x,Mask)
%   Author: Qiang Ning
%   Created: 11/19/2015
%   Decription: apply a 3D mask to a 4D matrix x; then change the resulting
%   values into a col vector.
%   Interface:
%           y = oper3D_mask_4D_data(x,Mask)
%   Input: 
%           x--L1xL2xL3xN, numerical
%           Mask--L1xL2xL3, boolean
%   Output:
%           y--column vector

[L1, L2, L3, N, Nc] = size(x);
assert(size(Mask,1)==L1&&size(Mask,2)==L2&&size(Mask,3)==L3);
Mask        = repmat(Mask,[1 1 1 N Nc]);
y           = x(Mask(:));
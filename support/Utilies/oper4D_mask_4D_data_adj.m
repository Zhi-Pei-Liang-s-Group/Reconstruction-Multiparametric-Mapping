function y = oper4D_mask_4D_data_adj(x,Mask)
%   Author: Yudu Li
%   Created: 03/15/2019
%   Decription: adjoint of oper4D_mask_4D_data; zero pad a col vector into 
%   a matrix according to Mask.
%   Input: 
%           x--column vector([sum(Mask(:))xN]x1)
%           Mask--boolean L1xL2xL3
%   Output:
%           y--L1xL2xL3xN matrix
[L1,L2,L3,N] = size(Mask);
Nc          = int16(length(x(:))/(sum(Mask(:))));
y           = zeros(L1,L2,L3,N,Nc,'like',x);
Mask        = repmat(Mask,[1,1,1,1,Nc]);
y(Mask(:))  = x(:);
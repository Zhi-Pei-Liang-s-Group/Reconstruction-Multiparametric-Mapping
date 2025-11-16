function y = oper3D_mask_4D_data_adj(x,Mask,Nc)
%   Author: Qiang Ning
%   Created: 11/19/2015
%   Decription: adjoint of oper3D_mask_4D_data; zero pad a col vector into 
%   a matrix according to Mask.
%   Input: 
%           x--column vector([sum(Mask(:))xN]x1)
%           Mask--boolean L1xL2xL3
%   Output:
%           y--L1xL2xL3xN matrix

if varsNotexistOrIsempty('Nc')
        Nc  = 1;
end
    
[L1,L2,L3]  = size(Mask);
N           = length(x) / sum(Mask(:)) / Nc;
y           = zeros(L1,L2,L3,N,Nc,'like',x);
Mask        = repmat(Mask,[1 1 1 N Nc]);
y(Mask(:))  = x(:);
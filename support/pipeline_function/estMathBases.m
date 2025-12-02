function [U,S,V]  = estMathBases(s_xt,Mask)
% This script estimates the mathematical bases from the EPSI data within
% the mask
%
% Modified by Yudu Li on 02/23/2019:
%  Allow multi-coil as input, the bases are estimated from all coils
%

if varsNotexistOrIsempty('Mask')
    Mask = true(size_dims(s_xt,1:3));
end

[L1,L2,L3,~,Nc] = size(s_xt);
if size(Mask,1) ~= L1 || size(Mask,2) ~= L2 || size(Mask,3) ~= L3
   Mask     = imresize3d(Mask,size_dims(s_xt,1:3)) > 0;
end

M        = size(s_xt,4);

csrt     = zeros(sum(Mask(:)),M,Nc);

for i = 1:Nc
    csrt_tmp = oper3D_mask_4D_data(s_xt(:,:,:,:,i),Mask);
    csrt(:,:,i) = reshape(csrt_tmp,[],M);
end

csrt     = permute(csrt,[1,3,2]);
csrt     = reshape(csrt,[],M);
[U,S,V]  = svd(csrt,'econ');
V        = conj(V);


function sos = sumOfSqr(data,dim)
% ======================================================
% calculate sum of square
% ------------------------------------------------------
if(nargin<2)
    dim = length(size(data));
end
sos = sqrt(sum(abs(data).^2,dim));  
function s=gen_LP_signal(c, z, M, sft)
%   GENERATE LINEAR PREDICTABLE SIGNALS
%   Author: Qiang Ning
%   Last Update: Apr. 23, 2014
%   s = gen_LP_signal(c, z, M, [sft])
%   Input: c, z, M, sft(default: 0)
%   Output: s
%   Expression: s(m)=\sum_{k=1}^K ck * zk ^ (m-1-sft), m=1,2,...,M
if length(c) ~= length(z)
    error('Dimension mismatch.');
end
if nargin < 4
    sft = 0;
end
s = zeros(M, 1);
for m = 1 : M
    for k = 1 : length(c)
        s(m) = s(m) + c(k)*z(k)^(m-1-sft);
    end
end

% @@ Yibo: this function is equivalent to
% s = sum(bsxfun(@times,bsxfun(@power,z,[0:M-1]).',a.'),2);
%   = bsxfun(@power,z,[0:M-1]-sft).'*c;
%
% @@ where the temporal basis is:
% Vt = bsxfun(@power,z,[0:M-1]-sft).';
%
% @@ and invididual LP components are:
% s_ind = bsxfun(@times,bsxfun(@power,z,[0:M-1]-sft).',a.');
%
% @@ how to contruct z from T2 and df:
% z = exp(1i*2*pi.*df*dt - 1./T2*dt);
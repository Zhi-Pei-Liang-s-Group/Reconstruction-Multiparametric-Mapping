function [z,f,T2,a,S]= mc_hsvd(xt, n, dt, L)
% HSVD
% Help doc by Qiang (May 21, 2014):
% [z, f, T2, a] = MC_HSVD(xt, n, dt)
% Author: Chao Ma
% Inputs
%       xt: a time-domain signal
%       n: number of most dominant sing. values to keep; MUST <=N, model
%          order.
%       dt: sampling period:
%       L: dimension of the hankel matrix, N/2 (N=length(xt)) by default 
% Outputs
%       z: z(j) = exp(1i*2*pi*f(j) - 1/T2(j))
%       a: linear coefficients of z
% Reference: H. Barkhuijsen, R. D. Beer, and D. V. Ormondt. Improved 
% algorithm for noniterative time-domain model fitting to exponentially 
% damped magnetic resonance signals. Journal of Magneic Resonance, pages 
% 553--557, 1987.

N = length(xt);
if nargin < 4
    L = round(N/2);
% 	L = round(N)/2;
end

% Form a hankel matrix 
c=xt(1:(L+1),1).';
r=xt((L+1):N,1).';
X=hankel(c,r);

% take SVD 
[U,S,V] = svd(X);

% discarding noise subspace: leave only n_singval principal sing. values/vectors
U=U(:,1:n);
V=V(1:n,:);

% finding U1=U_ - omitting the last row of U containing N largest sing.
% vectors
g=size(U,1);
U1=U(1:g-1,:);
 
% finding U2=U- - omitting the first row of U containing N largest sing.
% vectors
U2=U(2:g,:); 

z=eig(pinv(U1)*U2);

% Sorting z by angle(z)
[~,order]=sort(angle(z));                      
z=z(order);
f = angle(z)/(dt*2*pi); %   Should it be -f?
T2 = -dt./log(abs(z(:)));

Z=ones(length(xt),length(z));
temp=reshape(z, 1, length(z));
for i=2:length(xt) %generate Vandermonde matrix B based on poles
    Z(i,:)=Z(i-1,:).*temp;
end

if nargout>3
%     a=Z\xt;
    a = pinv(Z)*xt;
end

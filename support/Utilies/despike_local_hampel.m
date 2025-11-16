function I_out = despike_local_hampel(I, win, T, Mask)
% I: 3D image (double), win: [wx wy wz] odd sizes, T: threshold (e.g., 6)
if nargin < 2, win = [3 3 3]; end
if nargin < 3, T = 6; end
I = double(I);
pad = floor((win-1)/2);

% Local median
Imed = medfilt3(I, win);
Imed(isnan(Imed)) = 0;

% Local MAD (median absolute deviation)
AbsDev = abs(I - Imed);
AbsDev(AbsDev>1e-3) = 1e-3;
AbsDev_mean = mean(oper3D_mask_4D_data(AbsDev,Mask));
% MAD   = medfilt3(AbsDev, win);  % robust local scale proxy
% sigma = 1.4826 * MAD + 1e-8;

% Outlier mask (high outliers only)
% mask = (I - Imed) ./ sigma > T;
mask = abs(I - Imed) > T*AbsDev_mean;

% Replace outliers with local median
I_out = I;
I_out(mask) = Imed(mask);
end

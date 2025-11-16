function [ data ] = F3_x2k( image, dont_shift )
%F3_X2K Converts 3D spatial-spectral data from image space to data space.
%
% image         The spatial-spectral image. First 3 dims should be image space.
%               The 4th dim can be time or frequency.
% 
% dont_shift    (Optional) If true, don't do any fft shifts.
%--------------------------------------------------------------------------
%   Qiang 05/23/2016: split fft's into 3 lines

    if nargin < 2
        dont_shift = false;
    end
    
    [Ny, Nx, Nz, ~] = size(image);
    if dont_shift
        %data = sqrt(1/(Nx*Ny*Nz))*fft(fft(fft(image,[],1),[],2),[],3);
        data = fft(image,[],1);
        data = fft(data,[],2);
        data = fft(data,[],3)/sqrt(Nx*Ny*Nz);
    else
        data = sqrt(1/(Nx*Ny*Nz))*  fftshift(fft(ifftshift(...
                                        fftshift(fft(ifftshift(...
                                            fftshift(fft(ifftshift(...
                                                image,...
                                            1),[],1),1),...
                                        2),[],2),2),...
                                    3),[],3),3);
    end
    return;

end


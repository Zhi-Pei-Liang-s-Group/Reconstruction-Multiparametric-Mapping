function [ image ] = F3_k2x( data, dont_shift )
%F3_K2X Converts 3D spatial-spectral data from data space to image space.
%
% image         The spatial-spectral image. First 3 dims should be data space.
%               The 4th dim can be time or frequency.
% 
% dont_shift    (Optional) If true, don't do any fft shifts.
%--------------------------------------------------------------------------
%   Qiang 05/23/2016: split ifft's into 3 lines

    if nargin < 2
        dont_shift = false;
    end

    [Ny, Nx, Nz, ~] = size(data);
    if dont_shift
        %image = sqrt(Nx*Ny*Nz)*ifft(ifft(ifft(data,[],1),[],2),[],3);
        image = ifft(data,[],1);
        image = ifft(image,[],2);
        image = ifft(image,[],3)*sqrt(Nx*Ny*Nz);
    else
        image = sqrt(Nx*Ny*Nz)* fftshift(ifft(ifftshift(...
                                    fftshift(ifft(ifftshift(...
                                        fftshift(ifft(ifftshift(...
                                            data,...
                                        1),[],1),1),...
                                    2),[],2),2),...
                                3),[],3),3);    
    end
    return;

end

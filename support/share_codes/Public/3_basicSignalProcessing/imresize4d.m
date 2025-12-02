function [ c4d_up, orig_coords, new_coords ] = imresize4d( c4d, newDim, meth, threshold )
% 3D Interpolation, repeat for other dimensions
% ---- INPUTS ----
% c3d       : original image
% newDim    : new dimensions
% meth      : interpolation method [spline]
% threshold : threshold to generate binary output [0.5]
%
% ---- OUTPUTS ----
% c3d_up        : interpolated image
% orig_coords   : coordinates for the original image (used for interpolation)
% new_coords    : coordinates for the new image (used for interpolation)
%
% Author: Bryan A. Clifford
%   Created: 10-17-2014
% 	Updated: Chao Ma, 08-08-2015, Add threhold
%   Updated: Yibo Zhao, 05-24-2018, Support 4D data, support logical data
%
%   See also IMRESIZE3D
% ------------------------------------------------------------------------------

    %% ---- parse the inputs ----
    if nargin < 3
        meth = 'spline';
    end
    if nargin < 4
        threshold = 0.5;
    end

    dims    = size(c4d);
    if length(dims) == 2
        dims(3) = 1;
    end
    
    if length(newDim) == 2
        newDim(3) = 1;
    end
    
    try
        assert(length(newDim)<4);
    catch
        error('The new dimensions should be of length less or equal to 3');
    end
    
    %% ---- setting the grids and 3d interpolation----
    ny = (0:dims(1)-1) + .5;
    nx = (0:dims(2)-1) + .5;
    nz = (0:dims(3)-1) + .5;
    [gy_orig, gx_orig, gz_orig] = ndgrid(ny, nx, nz);
    
    ny_new = ( ( 0:newDim(1)-1 ) + .5 )*dims(1)/newDim(1);
    nx_new = ( ( 0:newDim(2)-1 ) + .5 )*dims(2)/newDim(2);
    nz_new = ( ( 0:newDim(3)-1 ) + .5 )*dims(3)/newDim(3);
    [gy_new, gx_new, gz_new] = ndgrid(ny_new, nx_new, nz_new);
    
    orig_coords.x = gx_orig;
    orig_coords.y = gy_orig;
    orig_coords.z = gz_orig;
    new_coords.x = gx_new;
    new_coords.y = gy_new;
    new_coords.z = gz_new;
    
    if ndims(c4d)<4
        % 3D
        if dims(3) > 1
            c4d_up = interp3(gx_orig, gy_orig, gz_orig, double(c4d), gx_new, gy_new, gz_new, meth);
        else
            c4d_up = interp2(gx_orig, gy_orig, double(c4d), gx_new, gy_new, meth);
        end
    else
        % 4D
        if dims(3) > 1
            if islogical(c4d)
                c4d_up = false([newDim,dims(4:end)]);
            else
                c4d_up = zeros([newDim,dims(4:end)],'like',c4d);
            end
            for more_ind = 1:prod(dims(4:end))
                c4d_up(:,:,:,more_ind) = interp3(gx_orig, gy_orig, gz_orig,...
                    double(c4d(:,:,:,more_ind)), gx_new, gy_new, gz_new, meth);
            end
        else
            if islogical(c4d)
                c4d_up = false([newDim,dims(4:end)]);
            else
                c4d_up = zeros([newDim,dims(4:end)],'like',c4d);
            end
            for more_ind = 1:prod(dims(4:end))
                c4d_up(:,:,more_ind) = interp2(gx_orig, gy_orig,...
                    double(c4d(:,:,more_ind)), gx_new, gy_new, meth);
            end
        end
    end
    
    if islogical(c4d)
        c4d_up = c4d_up >= threshold;
    end
    
    c4d_up = cast(c4d_up,'like',c4d);
    
end



function ims_out = imfill3D(ims_in,method)
    ims_out = ims_in;
    for nz = 1:size(ims_in,3)
        ims_out(:,:,nz) = imfill(ims_in(:,:,nz),method);
    end
    
    return;
end
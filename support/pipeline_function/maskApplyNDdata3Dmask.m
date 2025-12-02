function xtMat = maskApplyNDdata3Dmask(dataND,mask3D)
%% ======================================================================
% apply a 3D mask to ND data (x-y-z-t-c-a...) to form a matrix (x*y*z,t,c,a...)
% =======================================================================
% size check
if(any(size(dataND(:,:,:,1))~=size(mask3D(:,:,:,1))))
    warning('mask is resized to fit data');
    mask3D = imresize3d(mask3D,size(dataND(:,:,:,1)));
end
if(length(size(dataND))<=3)
    xtMat = dataND(mask3D);
    return;
end
NXm = sum(mask3D(:));
dims = size(dataND);
dimsRap = dims; 
dimsRap(1:3) = 1;
dimsMat = [NXm,dims(4:end)];
% apply mask
xtMat = zeros(dimsMat,'like',dataND);        % [X,t,c,a]
maskND = repmat(mask3D,dimsRap);             % [y,x,z,t,c,a]
xtMatN = dataND(maskND);
xtMat = reshape(xtMatN,size(xtMat));



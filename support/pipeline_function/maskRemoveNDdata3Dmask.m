function dataND = maskRemoveNDdata3Dmask(xtcMat,mask3D)
%% ======================================================================
% remove a 3D mask from a x-t-c... Matrix (x*y*z,t,c...) and recover the ND data (x-y-z-t-c...) 
% =======================================================================
% size check
dims3d      = size(mask3D);                 % [y,x,z]
dims        = size(xtcMat);                 % [X,t,c,a,...]
dataType    = class(xtcMat);
NXm         = sum(mask3D(:));
assert(NXm == size(xtcMat,1));
dimsDat     = [dims3d,dims(2:end)];         % [y,x,z,t,c,a...]
dimsRap     = [1,1,1,dims(2:end)];          % [y,x,z,t,c,a...]
% remove mask
dataND      = zeros(dimsDat,dataType);      % [y,x,z,t,c,a]
maskND      = repmat(mask3D,dimsRap);       % [y,x,z,t,c,a]
dataND(maskND) = xtcMat(:);

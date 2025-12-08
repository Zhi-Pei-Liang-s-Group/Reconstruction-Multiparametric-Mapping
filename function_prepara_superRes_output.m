function [Map_SR] = function_prepara_superRes_output( ...
    dataPath_LR, dataPath_SR, high_res_xy, value_range, paramTag)
% Helper function to assemble one parameter (e.g., T1 / PD / T2)
%
% Inputs
%   dataPath_LR : path to LR 2D .mat files
%   dataPath_SR : path to SR 2D .mat files for this parameter
%   high_res_xy : [Ny, Nx]
%   value_range : normalization scale
%
% Outputs
%   Map_LR : normalized LR map (Ny x Nx x Nz)
%   Map_SR : normalized SR map (Ny x Nx x Nz)
%   Map_HR : normalized HR map (Ny x Nx x Nz)
    
    totalSlide   = length(dir(dataPath_LR))-2;
    Map_SR = zeros(high_res_xy(1), high_res_xy(2), totalSlide, 'single');

    for indZ = 1:totalSlide

        % Load brainMask
        lrFile = fullfile(dataPath_LR, sprintf('2D_brain_%d.mat', indZ - 1));
        mask2d = myLoadVars(lrFile, 'brainMask');

        % Load SR result for this parameter
        srFile   = fullfile(dataPath_SR, sprintf('2D_brain_%d_0.mat', indZ - 1));
        superRes = myLoadVars(srFile, 'superRes');

        % Normalization, then apply mask
        Map_SR(:,:,indZ) = normalize_toAB(superRes, value_range(1), value_range(2)) .* mask2d;

        if mod(indZ, 10) == 0 || indZ == totalSlide
            fprintf('  [%s] slice %d / %d\n', paramTag, indZ, totalSlide);
        end
    end
end
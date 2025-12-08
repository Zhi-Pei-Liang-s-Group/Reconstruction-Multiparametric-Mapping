function function_prepara_superRes_input(T1map_3D, PDmap_3D, T2map_3D, ...
                                         subjectID, dataset_name, ...
                                         procDatSRPath, ...
                                         high_res_xy)
% FUNCTION_PREPARA_SUPERRES_INPUT
% Prepare 3D low-resolution T1 / PD / T2 maps as super-resolution network
% input. For each map, the function:
%   1) Clips the physical range to a predefined window
%   2) Builds a brain mask using non-zero voxels and fills small holes
%   3) Resamples the 3D map to a high-resolution in-plane canvas
%   4) Normalizes the map to [0, 1] within the brain mask
%   5) Saves each slice as 2D .mat files (lowRes, highRes, brainMask)
%   6) Saves PNG figures for quick visualization (lowRes vs highRes)
%
% INPUTS
%   T1map_3D       : 3D low-resolution T1 map (can be [], then it is skipped)
%   PDmap_3D       : 3D low-resolution PD map (can be [], then it is skipped)
%   T2map_3D       : 3D low-resolution T2 map (can be [], then it is skipped)
%   subjectID      : subject ID string, e.g. 'MRx_uploaded_demo_data'
%   dataset_name   : dataset label string, e.g. 'SPICE_352x352_from_214x122'
%   procDatSRPath  : root path for Water SR, (e.g. '/.../Water_T1T2_SR')
%   high_res_xy    : [Ny_HR, Nx_HR] for the HR canvas (optional, default [352 352])
%
% OUTPUTS
%   This is a utility function with no direct outputs. It writes:
%     - 2D_brain_*.mat files under:
%         <savePath_*>/Testing_data/<subjectID>/<dataset_name>/LR_data/mat
%     - panel_LR_HR_*.png under:
%         <savePath_*>/Testing_data/<subjectID>/<dataset_name>/LR_data/image
%
% DEPENDENCIES (should be in MATLAB path beforehand)
%   - imfill3D
%   - imresize3d
%   - imresize_ktrunc
%   - normalize_toAB
%   - mkdir_Yue
%
% Liang's Lab @ UIUC
% Created  : 2025-12-08
% Author   : Ziwen Ke
%

%% -------------------- Default HR canvas --------------------
if nargin < 7 || isempty(high_res_xy)
    % Default HR canvas (0.68 x 0.68 mm -> 352 x 352)
    high_res_xy = [352, 352];
end

fprintf('==== Prepare super-resolution inputs for subject %s, dataset %s ====\n', ...
        subjectID, dataset_name);

%% -------------------- T1 map processing --------------------
if ~isempty(T1map_3D) 
    fprintf('--- Preparing T1 map for super-resolution input ---\n');

    % T1-specific clipping range and display colormap
    value_range_T1 = [0.1, 3.0];
    cmap_T1        = 'jet';
    savePath_T1    = fullfile(procDatSRPath, 'T1map_SR');

    prepare_single_map(T1map_3D, ...
                       subjectID, dataset_name, ...
                       savePath_T1, high_res_xy, ...
                       value_range_T1, cmap_T1, ...
                       'T1');
end

%% -------------------- PD map processing --------------------
if ~isempty(PDmap_3D) 
    fprintf('--- Preparing PD map for super-resolution input ---\n');

    % PD-specific clipping range and display colormap
    value_range_PD = [0.0, 0.003];
    cmap_PD        = 'gray';
    savePath_PD    = fullfile(procDatSRPath, 'PDmap_SR');

    prepare_single_map(PDmap_3D, ...
                       subjectID, dataset_name, ...
                       savePath_PD, high_res_xy, ...
                       value_range_PD, cmap_PD, ...
                       'PD');
end

%% -------------------- T2 map processing --------------------
if ~isempty(T2map_3D) 
    fprintf('--- Preparing T2 map for super-resolution input ---\n');

    % T2-specific clipping range and display colormap
    value_range_T2 = [0.0, 0.3];
    cmap_T2        = 'hot';
    savePath_T2    = fullfile(procDatSRPath, 'T2map_SR');

    prepare_single_map(T2map_3D, ...
                       subjectID, dataset_name, ...
                       savePath_T2, high_res_xy, ...
                       value_range_T2, cmap_T2, ...
                       'T2');
end

fprintf('==== Done: super-resolution input preparation finished. ====\n');

end % end of function_prepara_superRes_input


%% ======================================================================
%% Internal helper: prepare one biomarker map
%% ======================================================================
function prepare_single_map(lowRes_map_3D, ...
                            subjectID, dataset_name, ...
                            savePath_root, high_res_xy, ...
                            value_range, cmap_name, biomarker_name)
% PREPARE_SINGLE_MAP
% Internal helper for function_prepara_superRes_input.
% Performs clipping, masking, HR resampling, normalization, and slice saving.

% -------------------- Create output directories --------------------
savePath_HR  = fullfile(savePath_root, 'Testing_data', subjectID, ...
                        dataset_name, 'LR_data', 'mat');
savePath_fig = fullfile(savePath_root, 'Testing_data', subjectID, ...
                        dataset_name, 'LR_data', 'image');

mkdir_Yue(savePath_HR);
mkdir_Yue(savePath_fig);

fprintf('  Output .mat directory : %s\n', savePath_HR);
fprintf('  Output image directory: %s\n', savePath_fig);

% -------------------- Basic size info --------------------
[Ny, Nx, L3] = size(lowRes_map_3D);
fprintf('  Input volume size for %s: [%d x %d x %d]\n', biomarker_name, Ny, Nx, L3);

% -------------------- Initial clipping --------------------
vmin = value_range(1);
vmax = value_range(2);

lowRes = lowRes_map_3D;
lowRes(lowRes < vmin) = vmin;
lowRes(lowRes > vmax) = vmax;

% -------------------- Brain mask (from non-zero voxels) --------------------
brainMask = imfill3D(lowRes ~= 0, 'holes');

% Resample brain mask to HR canvas (same number of slices)
brainMask_HR = imresize3d(brainMask, [high_res_xy(1), high_res_xy(2), L3]);

% -------------------- HR resampling of low-resolution map --------------------
% Here "highRes_map" is used as a placeholder for HR label. In this demo
% we simply upsample the same map. In training, this can be replaced by a
% higher-resolution reference if available.
highRes_map = lowRes;

lowRes_HR  = abs(real(imresize_ktrunc(lowRes, ...
                             [high_res_xy(1), high_res_xy(2), L3], 0, 1)));
lowRes_HR(lowRes_HR < vmin) = vmin;
lowRes_HR(lowRes_HR > vmax) = vmax;
lowRes_HR = lowRes_HR .* brainMask_HR;

highRes_HR = abs(real(imresize_ktrunc(highRes_map, ...
                             [high_res_xy(1), high_res_xy(2), L3], 0, 1)));
highRes_HR(highRes_HR < vmin) = vmin;
highRes_HR(highRes_HR > vmax) = vmax;
highRes_HR = highRes_HR .* brainMask_HR;

% -------------------- Normalize to [0, 1] within brain mask --------------------
lowRes_HR  = normalize_toAB(lowRes_HR,  0, 1) .* brainMask_HR;
highRes_HR = normalize_toAB(highRes_HR, 0, 1) .* brainMask_HR;

% -------------------- Slice-wise saving --------------------
count = 0;
for indZ = 1:L3
    % Extract 2D slices
    lowRes_slice    = squeeze(lowRes_HR(:, :, indZ));
    brainMask_slice = squeeze(brainMask_HR(:, :, indZ));
    highRes_slice   = squeeze(highRes_HR(:, :, indZ));

    % Save to 2D_brain_<index>.mat
    count = count + 1;
    outFile = fullfile(savePath_HR, sprintf('2D_brain_%d.mat', count-1));

    lowRes  = lowRes_slice; 
    brainMask = brainMask_slice; 
    highRes = highRes_slice;   

    save(outFile, 'highRes', 'brainMask', 'lowRes');

    fprintf('  [%s] Saved slice %d to %s\n', biomarker_name, count-1, outFile);

    % -------------------- Figure visualization --------------------
    margin = 2;  % virtual pixel margin used to compute normalized positions
    [H, W] = size(lowRes_slice);
    whole_w = 2 * (W + margin) + margin; % 2 columns (lowRes, highRes)
    whole_h = 1 * (H + margin);          % 1 row

    x_bot = margin / whole_h;
    y_1   = margin / whole_w;
    y_2   = (2 * margin + W) / whole_w;
    w_fra = W / whole_w;
    h_fra = H / whole_h;

    fh = figure('Position', [100, 100, whole_w, whole_h]);

    % First subplot: low-resolution (upsampled) input
    ax1 = subplot('Position', [y_1, x_bot, w_fra, h_fra]);
    imshow(lowRes_slice, [0, 1], 'InitialMagnification', 'fit');
    axis off;
    colormap(ax1, cmap_name);

    % Second subplot: "high-resolution" reference (here same as upsampled)
    ax2 = subplot('Position', [y_2, x_bot, w_fra, h_fra]);
    imshow(highRes_slice, [0, 1], 'InitialMagnification', 'fit');
    axis off;
    colormap(ax2, cmap_name);

    outFig = fullfile(savePath_fig, ...
                      sprintf('panel_LR_HR_%s_%d.png', biomarker_name, count));
    saveas(fh, outFig);
    close(fh);
end

fprintf('  [%s] Done. Total saved slices: %d\n', biomarker_name, count);

end % end of prepare_single_map

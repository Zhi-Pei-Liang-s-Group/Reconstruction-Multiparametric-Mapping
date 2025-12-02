function [idx_new, flag] = IdxIncrements(idx, grid)
%   Author: Qiang Ning
%   Update: May 22, 2014
%   Interface: [idx_new, flag] = IDXINCREMENTS(idx, grid)
%   Input:  
%           idx: a column array representing the current number
%           grid: an array representing the maximum of each digit
%   Output:
%           idx_new: idx+1
%           flag: successful or not
%   Example: if grid = [2 2 3], idx = [1 1 3], then idx_new = [1 2 1] and
%   flag = true; if idx = [2 2 3], then idx_new = [0 0 0] and flag = false.
N = length(idx);
if ~iscolumn(idx)
    idx = idx.';
end
flag = false;
for ii = N : -1 : 1
    if idx(ii) < grid(ii)
        idx_new = [idx(1:ii-1); idx(ii)+1; ones(N-ii, 1)];
        flag = true;
        break;
    end
end
if ~flag
    idx_new = zeros(N, 1);
end
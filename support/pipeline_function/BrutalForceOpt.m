function [a_opt, cost] = BrutalForceOpt(costfun, a_search_set)
%   Author: Qiang Ning
%   Update: May 22, 2014
%   Interface: [a_opt, cost] = BRUTALFORCEOPT(costfun, a_search_set)
%   Input:  
%           costfun: function handle dependent on "a".
%           a_search_set: search set for "a"; cell array.
%   Output:
%           a_opt: optimal a
%           cost: costfun(a_opt)
N               = length(a_search_set); % # of parameters

grid            = zeros(N, 1); 
for ii = 1 : N
    grid(ii)    = length(a_search_set{ii});
end

idx             = ones(N, 1);
a_tmp           = zeros(N, 1);

flag            = true;
cost            = inf;
while flag
    for ii = 1 : N
        a_tmp(ii) = a_search_set{ii}(idx(ii));
    end
    tmp         = costfun(a_tmp);
    if tmp < cost
        cost    = tmp;
        idx_opt = idx;
    end
    [idx, flag] = IdxIncrements(idx, grid);
end

a_opt           = zeros(N, 1);
for ii = 1 : N
    a_opt(ii)   = a_search_set{ii}(idx_opt(ii));
end

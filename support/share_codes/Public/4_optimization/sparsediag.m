function A = sparsediag(x)
% a fast and efficient way to generate sparse diagonal matrix from a vector
% effectively, sparsediag(x) == sparse(diag(vec(x)))
%
% Created by  Yibo Zhao @ UIUC, 2019/06/23
    
    A = spdiags(vec(x),0,numel(x),numel(x));
    
end
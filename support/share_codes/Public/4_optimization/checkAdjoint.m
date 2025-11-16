function rel_err = checkAdjoint(A,AH,x_size,rel_tol,verbose)
%% ==================================================================
%CHECKADJOINT check whether two operators are adjoint
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-11-01
%
%   [INPUTS]
%   ---- Required ----
%   A                       forward operator (matrix or function handle)
%   AH                      adjoint operator (matrix or function handle)
%   x_size                  data size for the input of A
%
%   ---- Optional ----
%   rel_tol                 relative tolerance [1e-8]
%   verbose                 degree of verbosity [1]
%
%   [OUTPUTS]
%   rel_err                 relative L2 error
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/11/01
%
%   Formulation:
%       For Hilbert spaces H1 and H2, A: H1 -> H2, AH: H2 -> H1.
%       According to the definition, for all x in H1, y in H2, <x,AH*y> = <A*x,y>.
%
%--------------------------------------------------------------------------

%% ------ parse the input and check ------

    % parse verbose
    if ~exist('verbose','var') || isempty(verbose)
        verbose = 1;
    end
    
    % parse rel_tol
    if ~exist('rel_tol','var') || isempty(rel_tol)
        if verbose>1
            rel_tol = inf;
        else
            rel_tol = 1e-8;
        end
    end
    
    % determine whether A is a matrix or a function
    [atype_fwd,afun_fwd,afcnstr_fwd] = myiterchk(A);
    [atype_adj,afun_adj,afcnstr_adj] = myiterchk(AH);
    
    % check size
    if strcmp(atype_fwd,'matrix')&&strcmp(atype_adj,'matrix')
        assert(all(size_dims(A,[1,2])==size_dims(AH,[2,1])),'Forward and adjoint matrices must match in size!');
    end
    
    if strcmp(atype_fwd,'matrix')
        if ~exist('x_size','var') || isempty(x_size)
            x_size = [size_dims(A,2),1];
        else
            x_size = x_size(:).';
            assert(size_dims(A,2)==x_size(1),'Forward matrix and input vector must match in size!');
        end
    end
    
    if strcmp(atype_adj,'matrix')
        if ~exist('x_size','var') || isempty(x_size)
            x_size = [size_dims(AH,1),1];
        else
            x_size = x_size(:).';
            assert(size_dims(AH,1)==x_size(1),'Adjoint matrix and input vector must match in size!');
        end
    end
    
%% --- check adjoint operator ---

    x_temp      = randn(x_size);
    Ax          = myiterapp('mtimes',afun_fwd,atype_fwd,afcnstr_fwd,x_temp);
    
    y_temp      = randn(size(Ax));
    AHy         = myiterapp('mtimes',afun_adj,atype_adj,afcnstr_adj,y_temp);

    inner_prod1 = vec(AHy)'*vec(x_temp);
    inner_prod2 = vec(y_temp)'*vec(Ax);
    
    rel_err     = l2err(inner_prod1,inner_prod2);
    
%% --- display ---

    if rel_err>rel_tol
        error('Adjoint check failed! L2 error of inner products: %1.3e.',rel_err);
    end
    
    if verbose
        fprintf('Adjoint check passed! L2 error of inner products: %1.3e.\n',rel_err);
    end
    
end
    
    

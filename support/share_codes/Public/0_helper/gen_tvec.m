function tvec = gen_tvec(Nt,dt,t0) 
%% ==================================================================
%GEN_TVEC generate t-vector
% ===================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-12-01
%
%   [INPUTS]
%   ---- Required ----
%   N                       length of t-vector
%   dt                      interval t-vector
%
%   ---- Optional ----
%   t0                      first time point [0]
%
%   [OUTPUTS]
%   tvec                    t-vector
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/12/01
%
%--------------------------------------------------------------------------

%%

    if varsNotexistOrIsempty('t0')
        t0 = 0;
    end
    
    % accept (x-t) data as first argument: use 4-th dimension length
    if ndims(Nt)>=4
        Nt = size(Nt,4);
    end
    
    tvec = [0:Nt-1]*dt + t0;
    
end



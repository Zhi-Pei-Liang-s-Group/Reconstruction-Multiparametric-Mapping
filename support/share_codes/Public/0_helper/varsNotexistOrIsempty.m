function X = varsNotexistOrIsempty(varargin)
%% ======================================================================
%VARSNOTEXISTORISEMPTY return an array of flags, true for each input
%variable does not exist or is empty.
% =======================================================================
%   Author: Yibo Zhao @ UIUC
%   Created: 2018-11-24
%
%   [INPUTS]
%   ---- Required ----
%   varNames                names of variables to test, either a string or a cell of strings
%
%   [OUTPUTS]
%   X                       an array of flags for each variable
%
%   Change log:
%       Created by  Yibo Zhao @ UIUC, 2018/11/24
%
%--------------------------------------------------------------------------

%% --- parse inputs ---
    varNames = parseInputs(varargin{:});
    assert(iscellstr(varNames),'Input must be a string or a cell of strings!');
    
%% --- check ---
    X = false([1,length(varNames)]);
    for ii = 1:length(varNames)
        cmd       = sprintf('~exist(''%s'',''var'')',varNames{ii});
        X(ii)     = evalin('caller',cmd); % exist: X = 0; non-exist: X = 1.

        if ~X(ii) % if exist, check whether it is empty
            cmd   = sprintf('isempty(%s)',varNames{ii});
            X(ii) = evalin('caller',cmd); % not empty: X = 0; empty: X = 1.
        end
    end

end

function varNames = parseInputs(varargin)
    
    narginchk(1,inf);
    
    for ii = 1:nargin
        if ischar(varargin{ii})
            if ~exist('varNames','var')
                varNames             = {varargin{ii}}; 
            else
                varNames             = [varNames,varargin{ii}];  
            end
            
        elseif iscellstr(varargin{ii})
            if ~exist('varNames','var')
                varNames             = varargin{ii};
            else
                varNames             = [varNames,varargin{ii}{:}];
            end
            
        % unexpected case
        else
            warning('Input #%d has illegal data type. Please check.',ii);
        end
    end
    
end

%#ok<*AGROW>
%#ok<*CCAT1>
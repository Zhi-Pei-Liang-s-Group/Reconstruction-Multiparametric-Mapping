function pLen = printProgress(pLen,idx,Idx,pStr,digFmt)
%% ============================================================== 
% Print progress by input index
% Input: 
%   pLen:   length of string to delete
%   idx:    iteration index
%   Idx:    total index
%   pStr:   string to show
%   digFmt: digital format
% Output: 
%   pLen:   length of string to delete next iteration
% Example: 
%   pLen = 0;
%   for i = 1:100
%       pause(0.1)
%       pLen = printProgress(pLen,i,100,'- test: ');
%   end
% ---------------------------------------------------------------
if varsNotexistOrIsempty('digFmt')
    digFmt = '%0.1f';
end

if(nargin<4)
    pStr = ' - Progress: ';
end
prgStr = sprintf([pStr,digFmt,'%%\n'],idx/Idx*100);
fprintf([repmat('\b',1,pLen) '%s'],prgStr);
pLen   = numel(prgStr);

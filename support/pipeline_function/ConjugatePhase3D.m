function dxt_corr = ConjugatePhase3D(dxt,fm_hz,tvec,flag_warning)
%   Interface: dxt_corr = ConjugatePhase3D(dxt,fm_hz,tvec)
%   12/01/15: adapted from 2D version
%   01/20/16: avoid KT_Trunc_3D when same size (fm_hz better be the same
%   size as dxt; otherwise KT_Zeropad_3D and KT_Trunc_3D will take much
%   time)
%   01/26/16: avoid parfor loops
%   05/25/16: use bsxfun
%   05/26/16: fixed the crash when fm_hz has a smaller size than dxt does
%   05/30/18: apply the conjugate phase more gracefully
%   05/31/18: turn off the warning in default mode
%   07/02/19: accept scalar fm_hz
%   08/08/19: accept dt as the third input

    %% --- parse the inputs ---
    
    if nargin < 4
       flag_warning = 1; 
    end
    
    if isscalar(tvec)
        if flag_warning
            warning('Input ''tvec'' is a scalar... Trying to parse it as dt');
        end
        tvec = [0:size(dxt,4)-1]*tvec;
    end
    
    if(length(tvec)~=size(dxt,4))
        error('Number of time points and number of x-t data echoes inconsistent!');
    end

    if(ndims(fm_hz)~=3 && flag_warning>1)
        warning('Field map is not 3D.');
    end
    
    if ndims(fm_hz)>3 && size(fm_hz,4) == size(dxt,5)
        fm_hz = permute(fm_hz,[1,2,3,5,4]);
    end
    
    dataSize = size(dxt);
    
    if isscalar(fm_hz)
        if flag_warning
            warning('Field map is a scalar... Repmat to 3D');
        end
        fm_hz = fm_hz*ones(size_dims(dxt,1:3));
    end
    
    if(~all( size_dims(dxt,1:3) == size_dims(fm_hz,1:3)) )
        oldDims = size_dims(dxt,1:3);
        newDims = max(size_dims(dxt,1:3),size_dims(fm_hz,1:3));
        warnmsg = sprintf(['Mismatching data size detected: ' ...
            '  x-t data: %d x %d x %d, field map: %d x %d x %d.' ...
            '\nResizing field map and data to %d x %d x %d. Please check size, or set flag_warning to 0.'], ...
            size(dxt,1),size(dxt,2),size(dxt,3),size(fm_hz,1), ...
            size(fm_hz,2),size(fm_hz,3),newDims(1),newDims(2),newDims(3));
        if flag_warning
            warning(warnmsg);
        end
        fm_hz   = imresize4d(fm_hz,newDims);
        if ~all(size_dims(dxt,1:3)==newDims)
            dxt     = imresize_ktrunc(dxt,newDims,false,false);
        end
    end

    %% --- conjugate phase ---
    phase_term  = bsxfun(@times,fm_hz,reshape(tvec,1,1,1,[]));
    dxt_corr    = bsxfun(@times,dxt,exp(-1i*2*pi*phase_term));

    %% --- package the output ---
    if exist('newDims','var')
        if flag_warning
            warning('Resizing data to the original size.');
        end
        if ~all(size_dims(dxt_corr,1:3)==oldDims)
            dxt_corr = imresize_ktrunc(dxt_corr,oldDims,false,false);
        end
    end
    dxt_corr    = reshape(dxt_corr,dataSize);
end


% archive: former version (before 05/30/18)
%{
[X,Y,Z] = size(fm_hz);
[x,y,z,M] = size(dxt);
if x~=X || y~=Y || z~=Z
    fm_hz = my3DImresize(fm_hz,max(x,X),max(y,Y),max(z,Z));
    dxt = KT_Zeropad_3D(dxt,max([x,y,z],[X,Y,Z]));
end
% dxt_corr = dxt .* ...
%            exp(-1i*2*pi*repmat(fm_hz,[1 1 1 M]).*repmat(permute(tvec,[2 3 4 1]),[X,Y,Z,1]));
dxt_corr = dxt.*exp(-1i*2*pi*bsxfun(@times,fm_hz,permute(tvec,[2 3 4 1])));
% dxt_corr = dxt;
% parfor l1 = 1 : X
%     for l2 = 1 : Y
%         for l3 = 1 : Z
%             dxt_corr(l1,l2,l3,:) = ...
%             squeeze(dxt_corr(l1,l2,l3,:)).*exp(-1i*2*pi*fm_hz(l1,l2,l3)*tvec);
%         end
%     end
% end
if x~=X || y~=Y || z~=Z
    dxt_corr = KT_Trunc_3D(dxt_corr,[x,y,z]);
end
%}

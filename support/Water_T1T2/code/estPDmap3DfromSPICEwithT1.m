function PDmap = estPDmap3DfromSPICEwithT1(sxt_spice,T1map,dt,FA,TR,Mask,flag_hsvd)
% This script estimates the PD map based on SPICE (or mGRE) data given T1
% map
%
% [Input]
%  -- sxt_spice: SPICE or mGRE data
%  -- T1map:     estimated T1 map
%  -- dt:        echo time for sxt_spice
%  -- FA:        flip angle, deg
%  -- TR:        repetition time, sec
%  -- Mask:      spatial support
%
% [Output]
%  -- PDmap:     estimated proton density map
%

if varsNotexistOrIsempty('flag_hsvd')
    flag_hsvd   = size(sxt_spice,4)>1;
end

%% estimated concentration from SPICE
if flag_hsvd
    [~,cfm,~,~] = hsvd_param(sxt_spice,[],dt,[-inf,inf],1,Mask);
    s0          = abs(cfm);
else
    s0          = abs(sxt_spice(:,:,:,1));
end

%% determine the PD map
alpha               = FA/180*pi;
PDmap               = s0.*((1-exp(-TR./T1map)*cos(alpha))./((1-exp(-TR./T1map))*sin(alpha)));
PDmap    = applySpatialSupport3d(PDmap,Mask);
PDmap(isnan(PDmap)) = 0;
PDmap(isinf(PDmap)) = 0;


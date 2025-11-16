function [x_prev,fval_iter] = linear_fitting_map_multi_mog_enforceReal(y,linear_A,Rank,Scale_cell,num_mog_cell,mix_mog_cell,mean_mog_cell,Sigma_mog_cell,sig_noise,options)
% linear fitting with MAP (MoG as prior)
% y:  input signal
% linear_A: linear forward model
% num_mog:  number of mog for each paramter
% mix_mog:  mixture density from mog: order from alpha to df
% mean_mog: mean from mog: order from alpha to df
% std_mog:  std from mog: order from alpha to df
% sig_noise: sigma of noise for real (or imaginary part)
% options: function options
%
% only support real number

options      = SetStructDefault(options, {'display','Tolx','TolFun','max_iter_mog'}, {'off',1e-5,1e-5,10});
flag_display = options.display;
Tolx         = options.Tolx; % nonlinsq
TolFun       = options.TolFun; % nonlinsq
max_iter_mog = options.max_iter_mog;

%% enforce real constraint
linear_A     = cat(1,real(linear_A),imag(linear_A));
y            = cat(1,real(y),imag(y));

%% update the MoG params due to scaling
Sigma_mog_inv_cell = Sigma_mog_cell;
num_mog_group = length(num_mog_cell); % number of different mog groups
for index_g = 1:num_mog_group
    
    % extract info for particular group
    Scale           = Scale_cell{index_g};
    Scale1          = Scale(1,:); % 1 x rank
    Scale2          = Scale(2,:); % 1 x rank

    % update
    num_mog         = num_mog_cell{index_g};
    
    mean_mog        = mean_mog_cell{index_g}; % num_mog x rank
    mean_mog        = bsxfun(@plus,bsxfun(@times,mean_mog,Scale1),Scale2); % num_mog x rank
    
    Sigma_mog       = Sigma_mog_cell{index_g}; % rank x rank x num_mog 
    Sigma_mog_inv   = Sigma_mog;
    for index_mog = 1:num_mog
        weights_scale1 = diag(vec(Scale1));
        Sigma_mog(:,:,index_mog) = weights_scale1*Sigma_mog(:,:,index_mog)*weights_scale1;
        Sigma_mog_inv(:,:,index_mog) = inv(Sigma_mog(:,:,index_mog));
    end
    
    % save
    Sigma_mog_cell{index_g}     = Sigma_mog;
    Sigma_mog_inv_cell{index_g} = Sigma_mog_inv;
    mean_mog_cell{index_g}      = mean_mog;
    
end

%% iteration solving map
opt_quad     = optimset('display',flag_display,'Tolx',Tolx,'TolFun',TolFun);
x_prev       = pinv(linear_A)*y;
fval_iter    = zeros(max_iter_mog,1);

for index_iter = 1:max_iter_mog
    
    H               = cell(num_mog_group,1);
    f               = cell(num_mog_group,1);
    
    %% derive operator for multi-MoG
    Ind             = 1;
    for index_g = 1:num_mog_group
        
        num_mog         = num_mog_cell{index_g};
        mean_mog        = mean_mog_cell{index_g};
        Sigma_mog_inv   = Sigma_mog_inv_cell{index_g};
        Sigma_mog       = Sigma_mog_cell{index_g};
        mix_mog         = mix_mog_cell{index_g};
        
        % extract current nonlin
        x_center_prev   = x_prev(Ind:Ind+Rank(index_g)-1).'; % 1 x rank
        x_center_prev   = repmat(x_center_prev,[num_mog,1]); % num_mog x rank
        x_center_prev   = (x_center_prev - mean_mog).'; % rank x num_mog
        
        % obtain the adaptive weight (stablized softmax)
        weights_tmp   = zeros(num_mog,1);
        for i = 1:num_mog
            weights_exp = x_center_prev(:,i);% rank x 1
            weights_exp = -0.5*weights_exp'*Sigma_mog_inv(:,:,i)*weights_exp;
            weights_exp = weights_exp + log(mix_mog(i)*sqrt(abs(det(Sigma_mog(:,:,i)))));
            weights_tmp(i) = weights_exp;
        end
        weights_tmp   = weights_tmp - max(weights_tmp);
        weights_tmp   = exp(weights_tmp);
        weights_tmp   = weights_tmp/(sum(weights_tmp)+eps);
        
        % re-organize model for quadratic programming (term 1)
        H{index_g}    = 0;
        for i = 1:num_mog
            H{index_g} = H{index_g} + weights_tmp(i)*(sig_noise.^2)*Sigma_mog_inv(:,:,i);
        end
        
        % re-organize model for quadratic programming (term 2)
        f{index_g}    = 0;
        for i = 1:num_mog
            f{index_g} = f{index_g} - 2*weights_tmp(i)*(sig_noise.^2)*mean_mog(i,:)*Sigma_mog_inv(:,:,i);
        end
        
        % upodate ind
        Ind       = Ind + Rank(index_g);
        
    end
    
    %% derive operator for likelihood
    H             = blkdiag(H{:});
    H             = H + linear_A'*linear_A;
    H             = (H+H')/2;
    
    f             = cell2mat(f.');
    f             = f -2*y'*linear_A;
    
    %% optimization
    H             = 2*H; % compensate for 1/2 in quadprog        
    [x_prev,fval] = quadprog(double(H),double(f),[],[],[],[],[],[],[],opt_quad);
    fval_iter(index_iter) = fval;    
    
end




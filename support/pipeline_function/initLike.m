function data = initLike(data)
% initialize like, Yibo Zhao @ UIUC, 2018/11/28

    if islogical(data)
        data = false(size(data));
    else
        data = zeros(size(data),'like',data);
    end
    
end


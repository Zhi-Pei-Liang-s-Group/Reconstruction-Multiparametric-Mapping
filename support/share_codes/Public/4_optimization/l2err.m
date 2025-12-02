function e = l2err(x,y)

    if islogical(x)
        x = double(x);
    end
    
    if islogical(y)
        y = double(y);
    end
    
    e = norm(x(:) - y(:))/norm(x(:));
    return;

end
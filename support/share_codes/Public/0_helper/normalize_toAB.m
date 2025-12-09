function res = normalize_toAB(data, minValue, maxValue)

    max_data = max(data(:));
    min_data = min(data(:));
    
    scaleFactor = (maxValue-minValue) / double((max_data-min_data));
    
    res      = minValue + (data - min_data)*scaleFactor;

end

function out = function_standardise_range(base_data, data_for_range, d_k)
    range_d = range(data_for_range);
    out = (base_data - min(data_for_range) - range_d*0.5)/(d_k*range_d);
end
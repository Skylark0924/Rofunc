function [mn, cv] = GMR_simple(data_in, gmr_mean, gmr_sigma, in, out)

for i = 1:size(data_in, 2)
    mn(:, i) = gmr_mean(out) + gmr_sigma(out, in)/gmr_sigma(in, in) * (data_in(:, i) - gmr_mean(in)); 
    cv(:, :, i) = gmr_sigma(out, out) - gmr_sigma(out, in)/gmr_sigma(in, in) * gmr_sigma(in, out);
end
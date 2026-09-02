function chi = getBump(r, tau, delta)
    r_p = (r + tau) / delta;
    r_m = (tau - r) / delta;
    
    f_p = exp(-1 ./ max(r_p, eps));
    f_m = exp(-1 ./ max(r_m, eps));
    g_p = exp(-1 ./ max(1 - r_p, eps));
    g_m = exp(-1 ./ max(1 - r_m, eps));
    
    chi = (f_p .* f_m) ./ ((f_p + g_p) .* (f_m + g_m));
end
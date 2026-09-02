function maxW = getMinW(stack)
    minW = zeros(1, 3);
    for i1 = 1:3
        i2 = mod(i1, 3) + 1;
        i3 = mod(i1 + 1, 3) + 1;  
        D = [stack.B{i2}-stack.B{i1}, stack.B{i3}-stack.B{i1}];
        m = [eye(2), ones(2, 1)];
        m = [m, -m];
        Pk = D(:, 1:2) * m;
        Pl = D(:, 3:4) * m;
        tol = 1e-5;
        min_len_k = min(vecnorm(Pk, 2, 1));
        Pk_hex = Pk(:, vecnorm(Pk, 2, 1) < min_len_k + tol);
        
        min_len_l = min(vecnorm(Pl, 2, 1));
        Pl_hex = Pl(:, vecnorm(Pl, 2, 1) < min_len_l + tol);
        
        P = [Pk_hex, Pl_hex];
        
        r = max(vecnorm(P, 2, 1)) ;
    
        angles = sort(mod(atan2(P(2, :), P(1, :)), 2*pi));
        gap = max([diff(angles), angles(1) + 2*pi - angles(end)]); 
        
        minW(i1) = 0.5 * r *sec(gap/2);
    end
    maxW = max(minW);
end
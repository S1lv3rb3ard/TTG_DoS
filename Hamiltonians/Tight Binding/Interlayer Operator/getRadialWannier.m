function [V_0, V_3, V_6] = getRadialWannier(r, aG)
    lambda = [0.3155, -0.0688, -0.0083];
    chi = [0, 0.5212, 1.5206];
    kappa = [2.0010, 0, 1.5731];
    xi = [1.7543, 3.4692, 2.8764];
    
    rBar = r ./ aG;
    
    V_0 = lambda(1) .* exp(-xi(1) .* rBar.^2) .* cos(kappa(1) .* rBar);
    V_3 = lambda(2) .* rBar.^2 .* exp(-xi(2) .* (rBar - chi(2)).^2);
    V_6 = lambda(3) .* exp(-xi(3) .* (rBar - chi(3)).^2) .* sin(kappa(3) .* rBar);
end
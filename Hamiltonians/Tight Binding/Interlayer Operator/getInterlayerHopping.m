function interH = getInterlayerHopping(stack, N_r, r_max, mK, mQ, maxQ)
    % 1. Setup real-space quadrature and radial component of Wannier functions
    [r, w] = getGaussLegendre(N_r);
    r = 0.5*r_max*(r+1);
    w = 0.5*r_max*w;
    [V0, V3, V6] = getRadialWannier(r, stack.aG);
    
    % 2. Compute Hankel transform of radial components
    maxK = sqrt(2) * maxQ;
    ranK = linspace(0, maxK, mK);
    arg = ranK(:) .* r';
    
    H0 = besselj(0, arg) * (w .* r .* V0);
    H3 = besselj(3, arg) * (w .* r .* V3);
    H6 = besselj(6, arg) * (w .* r .* V6);
    
    % Interpolate the 1D Fourier transforms
    FH0 = griddedInterpolant(ranK, H0, 'spline');
    FH3 = griddedInterpolant(ranK, H3, 'spline');
    FH6 = griddedInterpolant(ranK, H6, 'spline');
    
    % 3. Setup momentum discretization grids
    ranQ = linspace(-maxQ, maxQ, mQ);
    [qX, qY] = ndgrid(ranQ);
    K = sqrt(qX.^2 + qY.^2);
    
    origin = K < 1e-15;
    K_safe = K;
    K_safe(origin) = 1.0; % Prevent division by zero at the origin
    
    A0 = FH0(K);
    A3 = FH3(K);
    A6 = FH6(K);
    
    % 4. Setup symmetrization angles and rotated grids
    angles = [0, -120, -240];
    qXhat = qX ./ K_safe;
    qYhat = qY ./ K_safe;
    qXhat(origin) = 0; 
    qYhat(origin) = 0;
    
    c = reshape(cosd(angles), 1, 1, 3);
    s = reshape(sind(angles), 1, 1, 3);
    Rq_X = c .* qXhat - s .* qYhat;
    Rq_Y = s .* qXhat + c .* qYhat;  
    
    % 5. Compute interpolated hopping functions
    prefactor = stack.V^2/(6*pi);
    tau = stack.tau ./ vecnorm(stack.tau, 2, 1);
    interH = cell(2, 2, 2);
    
    for j = 1:2
        arg1 = Rq_X .* tau(1, j) + Rq_Y .* tau(2, j);
        arg2 = Rq_X .* tau(1, j+1) + Rq_Y .* tau(2, j+1);      
        % Inline evaluation of the 3rd Chebyshev polynomial (4x^3 - 3x)
        T312 = 4 * arg1.^3 - 3 * arg1;
        T321 = 4 * arg2.^3 - 3 * arg2;
        
        T6 = (2 * T312.^2 - 1) + (2 * T321.^2 - 1);
        
        for a = 1:2
            for b = 1:2
                T3 = ((-1)^(a+2) * T312) + ((-1)^(b+1) * T321);
                hoppings = prefactor .* sum(A0 + 1i.*A3.*T3 - A6.*T6, 3);
                interH{j, a, b} = griddedInterpolant({ranQ, ranQ}, hoppings, 'spline');
            end
        end
    end
end
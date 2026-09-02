function stack = getStack(aG, theta)
    stack.theta = theta;
    nTheta = size(stack.theta, 2);
    stack.aG = aG; % Graphene lattice constant
    A_G = stack.aG*[1 1/2; 0 sqrt(3)/2]; % Bravais lattice
    B_G = 2*pi*inv(A_G).'; % Reciprocal lattice
    tau_B = A_G*[1/3; 1/3]; % B site position
    K_G = B_G*[2/3; 1/3]; % K-point
    pK_G = B_G*[1/3; 2/3]; % K'-point
    
    stack.V = sqrt(abs(det(B_G)));
    stack.A = cell(1, nTheta);
    stack.B = cell(1, nTheta);
    stack.K = zeros(2, nTheta);
    stack.pK = zeros(2, nTheta);
    stack.tau = zeros(2, nTheta);
    
    for j = 1:nTheta
        R = rotd(stack.theta(j));
        stack.A{j} = R*A_G;
        stack.B{j} = R*B_G;
        stack.K(:, j) = R*K_G;
        stack.pK(:, j) = R*pK_G;
        stack.tau(:, j) = R*tau_B;
    end
    stack.tau = clean(stack.tau);
end
function [T, Q] = getTQ(wAA, wAB, K)
    phi = exp(1i*2*pi/3);
    cPhi = conj(phi);
    T = zeros(4, 3);
    T(:, 1) = [wAA; wAB; wAB; wAA];
    T(:, 2) = [wAA; wAB*cPhi; wAB*phi; wAA];
    T(:, 3) = [wAA; wAB*phi; wAB*cPhi; wAA];
    Q = cell(2, 1);
    for i1 = 1:2
        Q{i1}(:, 1) = K(:, i1)-K(:, i1+1);
        Q{i1}(:, 2) = rotd(-120)*Q{i1}(:, 1);
        Q{i1}(:, 3) = rotd(120)*Q{i1}(:, 1);
    end
end
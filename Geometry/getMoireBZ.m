function [MoireBZ, weights, gridIdx] = getMoireBZ(stack, N)
% GETMOIREBZ  Uniform N x N grid on the moire cell around K_j, one grid per X = 1, 2, 3.
%
%   Only the K valley is returned; its weight is doubled to account for K'.
%   This is exact for the truncated model: the DoF set is closed under G -> -G and
%   H(-q) = Pi*conj(H(q))*Pi with Pi: G -> -G, so the summed moments at -q equal those
%   at q to round-off. The K' contribution is therefore that of the grid -(K_j + C*x),
%   a representative of the K' valley equivalent to K'_j + C*x modulo R*_j. (The previous
%   version sampled K'_j + C*x; the two differ only at the level of the truncation error.)
%
%   gridIdx (2 x N^2 integers in 0..N-1) labels the points, so a grid of size N/d is the
%   subset with mod(gridIdx, d) == 0 -- see getDoS for the nested N sweep.

ranN = 0:N-1;
[I1, I2] = ndgrid(ranN);
gridIdx = [I1(:) I2(:)].';
X = gridIdx/N;
MoireBZ = cell(1, 3);
weights = zeros(1, 3);
for j = 1:3
    if j == 1 || j == 2
        C = stack.B{1}-stack.B{2};
    else
        C = stack.B{2}-stack.B{3};
    end
    weights(j) = 2*abs(det(C))/(N^2);          % factor 2: K' by time reversal
    MoireBZ{j} = C*X+stack.K(:, j);
end
end

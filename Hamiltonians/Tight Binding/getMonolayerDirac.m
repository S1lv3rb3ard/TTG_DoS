function [ED, hbarVF] = getMonolayerDirac(stack, tauIntra)
% GETMONOLAYERDIRAC  Dirac energy (eV) and hbar*v_F (eV*Angstrom) of the intralayer model.
%
%   Uses the same neighbour groups as getHamiltonianTB (getShells). With all
%   Jung-MacDonald Table II groups and a = 1.42*sqrt(3) A: ED = 3.17 meV and
%   hbar*vF = 5.577 eV*A. hbar*vF is averaged over 12 directions around K.
    if nargin < 2, tauIntra = Inf; end
    [Shells, W] = getShells(stack, tauIntra);
    tauB = stack.tau(:, 2);
    K = stack.K(:, 2);
    V1 = @(k) real(exp(1i*k.'*Shells{1})*W{1});
    V2 = @(k) exp(1i*k.'*tauB).*(exp(1i*k.'*Shells{2})*W{2});
    ED = V1(K);
    dk = 1e-5;
    angles = (0:11)*pi/6;
    v = zeros(size(angles));
    for n = 1:numel(angles)
        d = dk*[cos(angles(n)); sin(angles(n))];
        v(n) = abs(V2(K+d)-V2(K-d))/(2*dk);
    end
    hbarVF = mean(v);
end

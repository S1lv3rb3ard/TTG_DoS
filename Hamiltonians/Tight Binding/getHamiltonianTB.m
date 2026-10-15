function H = getHamiltonianTB(stack, DoF, tau, delta, tauIntra)
% GETHAMILTONIANTB  Reciprocal tight-binding Hamiltonian H(q) as a function handle.
%
%   H = getHamiltonianTB(stack, DoF, tau, delta[, tauIntra])
%     tau, delta  interlayer momentum cutoff (1/Angstrom). The Fourier-transformed
%                 interlayer hopping is multiplied by getBump(|Q|, tau, delta), which
%                 is 1 for |Q| <= tau - delta and falls smoothly to 0 at |Q| = tau.
%                 tau = 6.25 and delta = 0.25 give exactly RelaxedTTG's
%                 smoothCutoff(|Q|, 6.00, 6.25).
%     tauIntra    intralayer bond-length cutoff (Angstrom), applied to whole neighbour
%                 groups (getShells). Default Inf keeps all tabulated groups, as
%                 RelaxedTTG does.
%
%   Earlier versions used a single tau for both cutoffs, in different units (Angstrom
%   for the intralayer shells, 1/Angstrom for the interlayer bump).
    if nargin < 5, tauIntra = Inf; end
    intraSkeleton = getIntraSkeletonTB(stack, tauIntra, DoF);
    intraH =@(q) getIntralayerOperatorTB(q, stack, intraSkeleton);
    interSkeleton = getInterSkeletonTB(stack, DoF);
    interH =@(q) getInterlayerOperatorTB(q, tau, delta, DoF, interSkeleton);
    H =@(q) intraH(q)+interH(q);
end

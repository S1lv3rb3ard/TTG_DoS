function H = getHamiltonianBM(stack, DoF, opts)
% GETHAMILTONIANBM  Bistritzer-MacDonald continuum Hamiltonian H(q) on the DoF set.
%
%   H = getHamiltonianBM(stack, DoF[, opts]) with optional fields
%     opts.vf   hbar*v_F in eV*Angstrom. Default: the tight-binding monolayer value
%               from getMonolayerDirac (5.577 for Jung-MacDonald Table II), so the
%               continuum and tight-binding models share the Dirac velocity.
%               The earlier hard-coded value was 6.582119569*0.8 = 5.266.
%     opts.wAA, opts.wAB   interlayer tunnelling (eV); defaults 0.07 and 0.11.
%
%   q is the absolute momentum, as for getHamiltonianTB.
%
%   The construction below is built in reflected coordinates. It labels layer j's Dirac
%   point by the stored offset K_j - K_2 and selects hops by those offsets, so its
%   momentum q_c corresponds to the physical momentum 2*K_2 - q_c. The earlier version
%   evaluated it at q_c = q, which put the Dirac points of layers 1 and 3 at
%   2*K_2 - K_j (438 and 876 meV away from E = 0 at q = K_1, K_3) and mirrored the
%   continuum line cuts through K_2 relative to the tight-binding ones. Evaluating at
%   q_c = 2*K_2 - q places every layer's Dirac point at K_j. With matched v_F and
%   tunnelling, the 12 levels nearest E = 0 then agree with the tight-binding model to
%   4 meV at K_1, K_2 and K_3, against 8-13 meV before.
    if nargin < 3, opts = struct; end
    if ~isfield(opts, 'vf') || isempty(opts.vf)
        [~, opts.vf] = getMonolayerDirac(stack);
    end
    if ~isfield(opts, 'wAA'), opts.wAA = 0.07; end
    if ~isfield(opts, 'wAB'), opts.wAB = 0.11; end
    interH = getInterlayerOperatorBM(stack.K, DoF, opts.wAA, opts.wAB);
    intraH =@(qc) getIntralayerOperatorBM(qc-stack.K(:, 2), stack, DoF, opts.vf);
    H =@(q) clean(conj(interH+intraH(2*stack.K(:, 2)-q)));
end

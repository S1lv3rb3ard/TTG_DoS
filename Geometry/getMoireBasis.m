function basis = getMoireBasis(stack, DoF, W)
% GETMOIREBASIS  KPM start vectors for the total DoS, one set per q-grid X = 1, 2, 3.
%
%   By the shift identity (Eq. (4.55)), the diagonal of f(H_q) at a layer-l site with
%   reciprocal index G equals the layer-l centre diagonal at q + (sum of G), reduced mod R*_l.
%   Each grid X samples one layer over the W-disk, using the moire lattice C of its q-grid:
%
%       X = 1:  layer 3, sites G = ( B1*n, -B2*n, 0)   ->  centre at q + C12*n,  C12 = B1 - B2
%       X = 2:  layer 2, sites G = ( B1*n,  0,    0)   ->  centre at q + B1*n == q + C12*n (mod R*_2)
%       X = 3:  layer 1, sites G = ( 0,  B2*n, -B3*n)  ->  centre at q + C23*n,  C23 = B2 - B3
%
%   n runs over all integer pairs with |C*n| < W; n = 0 is that layer's centre site.
%
%   Previous version selected every site with G1+G2+G3 = C*n. For X = 1 and X = 3 that gives
%   the sets above, plus the centre sites of the other two layers. A layer-2 site has G2 = 0,
%   so its total G1+G3 contains no B2 and never equals C12*n or C23*n for n ~= 0. Layer 2 was
%   therefore sampled only at its centre, and X = 2 reused the X = 1 set (layer 3 again).
%   Layer 2 could equally use C23 with sites (0, 0, -B3*n), but its q-grid in getMoireBZ is
%   on C12, so C12 is used here.

B = stack.B;
mDoF = 2*size(DoF, 1);
layer = [3, 2, 1];
C = {B{1}-B{2}, B{1}-B{2}, B{2}-B{3}};

basis = cell(1, 3);
for j = 1:3
    limitN = floor(W*vecnorm(inv(C{j}), 2, 2));
    [N1, N2] = ndgrid(-limitN(1):limitN(1), -limitN(2):limitN(2));
    N = [N1(:) N2(:)]';
    N = N(:, vecnorm(C{j}*N, 2, 1) < W);

    target = zeros(size(N, 2), 6);              % rows: [G1 G2 G3] of the wanted sites
    switch j
        case 1, target(:, 1:2) = ( B{1}*N).';  target(:, 3:4) = (-B{2}*N).';
        case 2, target(:, 1:2) = ( B{1}*N).';
        case 3, target(:, 3:4) = ( B{2}*N).';  target(:, 5:6) = (-B{3}*N).';
    end

    rows = find(DoF(:, 9) == layer(j));
    mask = pdist2(target, DoF(rows, 3:8)) < 1e-9;
    [~, c] = find(mask);
    c = rows(sort(c));

    idx = 2*(c-1)+(1:2);                        % both orbitals of each site
    idx = idx.';
    idx = idx(:);
    basis{j} = sparse(idx, 1:numel(idx), 1, mDoF, numel(idx));
end
end

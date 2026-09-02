function basis = getMoireBasis(stack, DoF, W)
invB = cellfun(@inv, stack.B, 'uni', false);
invB = blkdiag(invB{:});
compI = round(invB*DoF(:, 3:8).');
P = cat(2, stack.B{:});
compG = P*compI;

basis = cell(1, 3);
for j = 1:3
    if j == 1
        C = stack.B{1}-stack.B{2};
    elseif j == 2
        basis{2} = basis{1};
        continue;
    elseif j == 3
        C = stack.B{2}-stack.B{3};
    end
    limitN = floor(W*vecnorm(inv(C), 2, 2));
    ranN1 = -limitN(1):limitN(1);
    ranN2 = -limitN(2):limitN(2);
    [N1, N2] = ndgrid(ranN1, ranN2);
    N = [N1(:) N2(:)]';
    G = C*N;
    validG = G(:, vecnorm(G, 2, 1)<W);

    mask = pdist2(validG.', compG.')<1e-15;
    [~, c] = find(mask);

    idx = 2*(c-1)+(1:2);
    idx = idx.';
    idx = idx(:);

    mBasis = numel(idx);
    mDoF = 2*size(DoF, 1);

    basis{j} = sparse(idx, 1:mBasis, 1, mDoF, mBasis);
end
end
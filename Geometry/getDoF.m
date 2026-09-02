function DoF = getDoF(stack, W, L, output)
B = stack.B;

%[~, idxK] = min(vecnorm(q-stack.K, 2, 1));
%qNew = q-stack.K(:, idxK);

minSigma = min(cellfun(@(x) min(svd(x)), B));
mMax = ceil(L/minSigma);
[m1, m2] = ndgrid(-mMax:mMax);
m = [m1(:), m2(:)]';

minSigma = inf;
for i1 = 1:3
    i2 = mod(i1, 3)+1;
    i3 = mod(i1+1, 3)+1;
    sigma = min(svd(B{i2}-B{i1}));
    minSigma = min(minSigma, sigma);
    sigma = min(svd(B{i3}-B{i1}));
    minSigma = min(minSigma, sigma);
end
nMax = ceil(W/minSigma)+1;
[n1, n2] = ndgrid(-nMax:nMax);
n = [n1(:) n2(:)]';

mCountMax = (2*mMax+1)^2;
nCountMax = (2*nMax+1)^2;
maxCount = 6*mCountMax*nCountMax;

DoF = zeros(maxCount, 11);

shiftK = stack.K-stack.K(:, 2);

idxShift = 0;
for i1 = 1:3
    i2 = mod(i1, 3)+1;
    i3 = mod(i1+1, 3)+1;
    normL1 = sqrt(sum((B{i2}*m).^2, 1));
    mask = normL1<L;
    C = B{i2}-B{i1};
    D = B{i3}-B{i1};
    numValid = nnz(mask);
    if numValid~=0
        normL1 = normL1(:, mask);
        x = m(:, mask);
        E = -inv(D)*C;
        for i4 = 1:nnz(mask)
            x0 = x(:, i4);
            center = round(E*x0);
            y = n+center;
            normL2 = sqrt(sum((B{i3}*y).^2, 1));
            mask = normL2<L;
            numValid = nnz(mask);
            if numValid~=0
                normL2 = normL2(mask);
                maxNormL = max(normL1(i4), normL2);
                y = y(:, mask);
                k = C*x0+D*y;
                normW = sqrt(sum(k.^2, 1));
                mask = normW<W;
                numValid = nnz(mask);
                if numValid~=0
                    y = y(:, mask);
                    k = k(:, mask)+shiftK(:, i1);
                    maxNormL = maxNormL(mask);
                    normW = normW(mask);
                    idx = idxShift+(1:numValid);
                    idxShift = idxShift+numValid;
                    DoF(idx, 1:2) = k.';
                    DoF(idx, 2*i2+(1:2)) = repmat((B{i2}*x0).', numValid, 1);
                    DoF(idx, 2*i3+(1:2)) = (B{i3}*y).';
                    DoF(idx, 9) = i1;
                    DoF(idx, 10) = maxNormL.';
                    DoF(idx, 11) = normW.';
                end
            end
        end
    end
end

DoF = DoF(1:idxShift, :);
[DoF, ~] = sortrows(DoF, [9, 1:8]);

if strcmpi(output, 'clean')
    S = getSparsity(DoF, 'full');
    mask = sum(S, 2)>=2;
    DoF = DoF(mask, :);
end

end
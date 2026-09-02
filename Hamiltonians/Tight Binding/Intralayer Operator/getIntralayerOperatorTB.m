function intraH = getIntralayerOperatorTB(q, stack, intraSkeleton)
    m = sum(cellfun(@(x) size(x, 1), intraSkeleton.G));
    [v, r, c] = deal(zeros(4*m, 1));
    t = 0;
    offset = [1, 1, 2, 2; 1, 2, 1, 2];
    for i1 = 1:3
        mLayer = numel(intraSkeleton.r{i1});
        k = q'*intraSkeleton.R{i1}+intraSkeleton.G{i1};         
        I = intraSkeleton.r{i1};
        V{1} = real(exp(1i*k*intraSkeleton.Shells{1})*intraSkeleton.ShellWeights{1});
        V{2} = exp(1i*k*stack.tau(:, 2)).*exp(1i*k*intraSkeleton.Shells{2})*intraSkeleton.ShellWeights{2};
        V{3} = conj(V{2});
        V{4} = V{1};
        for i2 = 1:4
            idx = t+(1:mLayer);
            r(idx) = I+offset(1, i2);
            c(idx) = I+offset(2, i2);
            v(idx) = V{i2};
            t = t+mLayer;
        end
    end
    intraH = sparse(r, c, v, 2*m, 2*m);
end
function interH = getInterlayerOperatorTB(q, tau, delta, DoF, interSkeleton)
m = size(DoF, 1);
maxEntries = 4*sum(cellfun(@(x) size(x, 1), interSkeleton.G));
[r, c, v] = deal(zeros(maxEntries, 1));
offset = [1, 1, 2, 2; 1, 2, 1, 2];
t = 0;
for i1 = 1:2
    Q = q'+interSkeleton.G{i1};
    normQ = vecnorm(Q, 2, 2);
    w = getBump(normQ, tau, delta);
    mask = normQ<=tau & w>eps;
    mLayer = nnz(mask);
    Q = Q(mask, :);
    I = interSkeleton.r{i1}(mask);
    J = interSkeleton.c{i1}(mask);
    w = w(mask);
    for i2 = 1:4
        idx = t+(1:mLayer);
        r(idx) = I+offset(1, i2);
        c(idx) = J+offset(2, i2);
        v(idx) = w.*interSkeleton.P{i1}(mask, i2).*interSkeleton.h{i1, offset(1, i2), offset(2, i2)}(Q);
        t = t+mLayer;
    end
end
r = r(1:t);
c = c(1:t);
v = v(1:t);

interH = sparse(r, c, v, 2*m, 2*m);
interH = interH+interH';
end
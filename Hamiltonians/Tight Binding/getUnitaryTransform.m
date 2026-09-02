function U = getUnitaryTransform(tauVec, DoF) % To match Taylor expansion with BM model
    m = size(DoF, 1);
    v = ones(1, 2*m);
    v(2:2:end) = exp(1i*DoF(1:end, 3:8)*tauVec);
    U = sparse(1:2*m, 1:2*m, v, 2*m, 2*m);
end
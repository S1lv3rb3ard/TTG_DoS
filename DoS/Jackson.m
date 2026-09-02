function J = Jackson(P)
    %maxP = max(P);
    x = 0:P;
    c = cos(pi .* x ./ (P + 1));
    s = sin(pi .* x ./ (P + 1));
    t = cot(pi ./ (P + 1));
    J = ((P - x + 1) .* c + s .* t) ./ (P + 1);
    mask = x<P;
    J(~mask) = 0;
end
function [X, W] = getGaussLegendre(N)
    % Golub-Welsch Algorithm
    beta = 0.5 ./ sqrt(1 - (2*(1:N-1)).^-2);
    T = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(T);
    X = diag(D);
    [X, i] = sort(X);
    W = 2 * V(1, i).^2;
    W = W(:);
end
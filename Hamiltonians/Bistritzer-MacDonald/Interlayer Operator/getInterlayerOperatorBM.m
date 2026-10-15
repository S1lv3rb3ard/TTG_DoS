function interH = getInterlayerOperatorBM(K, DoF, wAA, wAB)
% wAA, wAB: AA and AB tunnelling (eV); defaults 0.07 and 0.11.
if nargin < 3 || isempty(wAA), wAA = 0.07; end
if nargin < 4 || isempty(wAB), wAB = 0.11; end
m = size(DoF, 1);
S = getSparsity(DoF, 'block');
% Compute hopping matrices and neighbor vector
[T, Q] = getTQ(wAA, wAB, K);
% Compute maximum number of nonzero entries and 
maxEntries = 2*nnz(S{1, 2})+2*nnz(S{2, 3});
[r, c, v] = deal(zeros(maxEntries, 1));
offset = [1, 1, 2, 2; 1, 2, 1, 2];
t = 0;
for i1 = 1:2
    % Set off layer counters
    i2 = mod(i1, 3)+1;
    % Get global Hamiltonian indices
    refI = find(DoF(:, 9) == i1);
    refJ = find(DoF(:, 9) == i2);
    % Get valid local Hamiltonian indices
    [I, J] = find(S{i1, i2});
    % Globalize local Hamiltonian Indices
    I = refI(I);
    J = refJ(J);
    % Get corresponding vectors
    k = DoF(I, 1:2)'-DoF(J, 1:2)';
    % Check if entry gets T(:, i3)
    mask = zeros(3, size(k, 2));
    tol = 1e-9;    % was 1e-15: exact float comparison of momentum differences
    for i3 = 1:3
        mask(i3, :) = all(abs(k-Q{i1}(:, i3))<=tol, 1);
    end
    mask = sparse(mask);
    % Check if entry gets any T
    valid = any(mask, 1);
    % Restrict computation to valid entries
    mValid = nnz(valid);
    I = 2*(I(valid)-1);
    J = 2*(J(valid)-1);
    % Compute 2x2 blocks and assign to global structure
    V = T*mask(:, valid);
    for i2 = 1:4
        idx = t+(1:mValid);
        r(idx) = I+offset(1, i2);
        c(idx) = J+offset(2, i2);
        v(idx) = V(i2, :);
        t = t+mValid;
    end
end
r = r(1:t);
c = c(1:t);
v = v(1:t);
interH = sparse(r, c, v, 2*m, 2*m);
interH = interH+interH';
end
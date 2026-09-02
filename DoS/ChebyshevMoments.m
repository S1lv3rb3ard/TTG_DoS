function Z = ChebyshevMoments(X, Y, P, mode, precision)
% CHEBYSHEVMOMENTS  KPM Chebyshev moments mu_n = <Y|T_n(H)|Y>, summed per q-point.
%
%   X         cell of (scaled) Hamiltonians, one per q-point in the batch (H_i = X{i})
%   Y         cell of basis selection vectors, one per q-point (Y_i = Y{i})
%   P         highest moment order (returns mu_0 ... mu_P, i.e. P+1 columns)
%   mode      'gpu' or 'cpu'
%   precision (optional) 'double' (default) or 'single' -- runs the recursion in
%             single precision (~2x less memory traffic; needs MATLAB with single
%             sparse support, i.e. R2025b+). Moments are returned as double either way.
%
% Two moments per matrix-vector product. Because the moments are the autocorrelation
% form mu_n = <Y|T_n(H)|Y> (same vector on both sides) and H is Hermitian, the standard
% KPM doubling identities hold:
%       mu_{2n}   = 2 <T_n  |T_n> - mu_0
%       mu_{2n+1} = 2 <T_{n+1}|T_n> - mu_1
% so all P+1 moments come from ~P/2 matvecs instead of P. This is algebraically
% identical to the one-moment-per-matvec recursion (matches it to floating point).

    % --- assemble the batch: block-diagonal H, vertically stacked basis vectors ---
    mIdx = numel(X);
    VecLength = cellfun(@(x) size(x, 1), X);
    S = arrayfun(@(x) ones(1, x), VecLength, 'uni', false);
    S = sparse(blkdiag(S{:}));            % mIdx x sum(VecLength): sums each q-point's rows
    X = sparse(blkdiag(X{:}));            % one block-diagonal Hamiltonian for the whole batch

    CellSize = cellfun(@(x) size(x, 2), Y);
    MaxCellSize = max(CellSize);
    CellIdx = find(CellSize ~= MaxCellSize);
    if ~isempty(CellIdx)
        Y(CellIdx) = cellfun(@(x) [x, zeros(size(x, 1), abs(size(x, 2)-MaxCellSize))], ...
                             Y(CellIdx), 'uni', false);
    end
    Y0 = cat(1, Y{:});                    % stacked initial basis vectors (0/1)

    if nargin >= 5 && strcmpi(precision, 'single')
        X = single(X); S = single(S); Y0 = single(Y0);   % run the hot path in single
    end

    Z = zeros(mIdx, P+1);                 % moments kept double (single results upcast on store)
    if strcmpi(mode, 'gpu')
        X  = gpuArray(X);
        S  = gpuArray(S);
        Y0 = gpuArray(Y0);
        Z  = gpuArray(Z);
    end
    Y0 = full(Y0);

    % per-q-point inner product  <A|B> = sum over each q-point's rows of Re(conj(A).*B)
    ip = @(A, B) S * sum(real(conj(A) .* B), 2);

    % --- base moments mu_0, mu_1 ---
    Tprev = Y0;            % T_0 = Y0
    Tcur  = X * Y0;        % T_1 = H * Y0   (X not yet scaled by 2)
    mu0 = ip(Tprev, Tprev);                 % mu_0 = <T0|T0>
    mu1 = ip(Tcur,  Tprev);                 % mu_1 = <T1|T0>
    Z(:, 1) = mu0;
    if P >= 1
        Z(:, 2) = mu1;
    end

    % --- two moments per matvec ---
    X = sparse(2 .* X);                             % the recursion uses 2H
    for n = progress(1:floor(P/2))
        Z(:, 2*n+1) = 2*ip(Tcur, Tcur) - mu0;       % mu_{2n}
        Tnext = X*Tcur - Tprev;                     % T_{n+1}  (the single matvec)
        if 2*n+2 <= P+1
            Z(:, 2*n+2) = 2*ip(Tnext, Tcur) - mu1;  % mu_{2n+1}
        end
        Tprev = Tcur;
        Tcur  = Tnext;
    end
end

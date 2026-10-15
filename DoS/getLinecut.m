function linecut = getLinecut(K, N_q, gammaConvention)
% GETLINECUT  The K1-K2-Gamma12-Gamma23-K2-K3 path with N_q points per segment.
%
%   Gamma_jk is the centre of the moire Brillouin zone of the interface (j,k): the
%   hexagon with adjacent corners K_j and K_k. It lies on the perpendicular bisector
%   of K_j K_k, at distance (sqrt(3)/2)|K_k - K_j| from the midpoint, so
%   |Gamma_jk - K_j| = |Gamma_jk - K_k| = |K_k - K_j|.
%
%   gammaConvention = 'legacy' reproduces the earlier construction, used for the line
%   cuts of arXiv:2606.13434 Figs. 4-5: distance |K_k - K_j| from the midpoint, which
%   is 1.118|K_k - K_j| from the corners and not the moire zone centre.
%   Identical to RelaxedTTG's rttg_common.getLinecut.
    if nargin < 3 || isempty(gammaConvention), gammaConvention = 'moire'; end
    M = (K(:, 1:2)+K(:, 2:3))/2;
    v = K(:, 2:3)-K(:, 1:2);
    w = rotd(-90)*v;
    w = w./vecnorm(w, 2, 1);
    switch lower(gammaConvention)
        case 'moire'
            C = M+w.*(sqrt(3)/2).*vecnorm(v, 2, 1);
        case 'legacy'
            C = M+w.*vecnorm(v, 2, 1);
        otherwise
            error('getLinecut:gammaConvention', 'gammaConvention must be ''moire'' or ''legacy''.');
    end
    Q = [K(:, 1:2), C, K(:, 2:3)];
    N_Q = size(Q, 2);
    linecut.q = Q(:, 1);
    linecut.tickLocs = zeros(1, N_Q);
    linecut.kPath = 0;
    for j = 1:(N_Q-1)
        pStart = Q(:, j);
        pEnd = Q(:, j+1);
        dist = norm(pEnd-pStart);
        X = [linspace(pStart(1), pEnd(1), N_q+1); ...
             linspace(pStart(2), pEnd(2), N_q+1)];
        currentDist = linecut.tickLocs(j);
        segDist = linspace(0, dist, N_q+1)+currentDist;
        linecut.q = [linecut.q, X(:, 2:end)];
        linecut.kPath = [linecut.kPath, segDist(2:end)];
        linecut.tickLocs(j+1) = currentDist+dist;
    end
    linecut.labels = {'$K_{1}$', '$K_{2}$', '$\Gamma_{12}$', '$\Gamma_{23}$', '$K_{2}$', '$K_{3}$'};
end

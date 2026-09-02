function linecut = getLinecut(K, N_q)
    M = (K(:, 1:2)+K(:, 2:3))/2;
    v = K(:, 2:3)-K(:, 1:2);
    w = rotd(-90)*v;
    w = w./vecnorm(w, 2, 1);
    C = M+w.*vecnorm(v, 2, 1);
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
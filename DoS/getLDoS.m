function Job = getLDoS(Job, E, mode)
g = Jackson(Job.P);
weight = 1/6;
uniqueTheta = sort(unique(Job.list.t));
numTheta = numel(uniqueTheta);
for i1 = 1:numTheta
    hereT = Job.list.t==i1;
    scale = Job.scale(i1);
    scaledE = scale*E;
    T = Chebyshev(Job.P, scaledE, mode);
    T(1, :) = 0.5 * T(1, :);
    ChebW = (scale * 2) ./ (pi * sqrt(1 - scaledE.^2));
    Job.list.DoS(hereT, :) = weight*((g.*Job.list.mu(hereT, :))*T) .* ChebW;
end
Job.list.DoS = gather(Job.list.DoS);
end
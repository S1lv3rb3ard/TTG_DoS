function Job = getDoS(Job, E, mode)
    g = Jackson(Job.P);
    uniqueTheta = sort(unique(Job.list.t));
    numTheta = numel(uniqueTheta);
    for i1 = 1:numTheta
        hereT = Job.list.t==i1;
        scale = Job.scale(i1);
        scaledE = scale*E;
        T = Chebyshev(Job.P, scaledE, mode);
        T(1, :) = 0.5 * T(1, :);
        ChebW = (scale * 2) ./ (pi * sqrt(1 - scaledE.^2));
        for i2 = 1:3
            hereX = hereT & (Job.list.X == i2);
            hereN = Job.list.N(hereX);
            weight = Job.weights{i1}(hereN, i2);
            Job.list.DoS(hereX, :) = weight.*((g.*Job.list.mu(hereX, :))*T).*ChebW;
        end
    end
    Job.list.DoS = gather(Job.list.DoS);
    Job.list = groupsummary(Job.list, ["t", "N", "model"], "sum", "DoS");
    Job.list.GroupCount = [];
    Job.list = renamevars(Job.list, 'sum_DoS', 'DoS');
end
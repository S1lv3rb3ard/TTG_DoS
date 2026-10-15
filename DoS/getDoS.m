function Job = getDoS(Job, E, mode, Puse)
% GETDOS  Reconstruct the DoS from stored Chebyshev moments.
%
%   Job = getDoS(Job, E, mode)        uses all Job.P moments.
%   Job = getDoS(Job, E, mode, Puse)  uses moments 0..Puse only (Puse <= Job.P). The first
%                                     Puse+1 moments of a P-moment run are exactly those of a
%                                     Puse-moment run, so one run at the largest P gives the
%                                     DoS for every smaller P (e.g. P = 1000*2.^(0:3)).
%
%   Nested N sweep: if the Job was built for a single grid Job.Nmax with the gridIdx stored
%   in Job.list.ix (see makeStage / testDoS), the DoS is returned for every N in Job.Nsweep
%   (each must divide Nmax). The N-grid is the subset mod(ix, Nmax/N) == 0 of the Nmax-grid,
%   so these are the same q-points, and the same moments, a separate N-grid run would compute.
%   In that case Job.list.N holds the actual N.
%
%   Optional Job.shift (default 0): moments were computed for scale*(H - shift).

    if nargin < 4 || isempty(Puse), Puse = Job.P; end
    assert(Puse <= Job.P, 'getDoS:Puse', 'Puse = %d exceeds the %d stored moments.', Puse, Job.P);
    g = Jackson(Puse);
    if isfield(Job, 'shift'), shift = Job.shift; else, shift = zeros(size(Job.scale)); end
    mu = Job.list.mu(:, 1:Puse+1);

    nested = isfield(Job, 'Nsweep') && ismember('ix', Job.list.Properties.VariableNames);
    uniqueTheta = sort(unique(Job.list.t));
    numTheta = numel(uniqueTheta);

    if ~nested
        % ---- original path: one row per (t, N, X, q); summed per (t, N, model) ----
        DoS = zeros(height(Job.list), numel(E));
        for i1 = 1:numTheta
            hereT = Job.list.t == i1;
            scale = Job.scale(i1);
            scaledE = scale*(E-shift(i1));
            T = Chebyshev(Puse, scaledE, mode);
            T(1, :) = 0.5 * T(1, :);
            ChebW = (scale * 2) ./ (pi * sqrt(1 - scaledE.^2));
            for i2 = 1:3
                hereX = hereT & (Job.list.X == i2);
                hereN = Job.list.N(hereX);
                weight = Job.weights{i1}(hereN, i2);
                DoS(hereX, :) = gather(weight.*((g.*mu(hereX, :))*T).*ChebW);
            end
        end
        Job.list.DoS = DoS;
        Job.list = groupsummary(Job.list, ["t", "N", "model"], "sum", "DoS");
        Job.list.GroupCount = [];
        Job.list = renamevars(Job.list, 'sum_DoS', 'DoS');
        return
    end

    % ---- nested path: sum the weighted moments per (t, N, model), then one transform ----
    Nmax = Job.Nmax;
    assert(all(mod(Nmax, Job.Nsweep) == 0), 'getDoS:Nsweep', 'Every N in Nsweep must divide Nmax = %d.', Nmax);
    models = unique(Job.list.model);
    out = table;
    for i1 = 1:numTheta
        hereT = Job.list.t == i1;
        scale = Job.scale(i1);
        scaledE = scale*(E-shift(i1));
        T = Chebyshev(Puse, scaledE, mode);
        T(1, :) = 0.5 * T(1, :);
        ChebW = (scale * 2) ./ (pi * sqrt(1 - scaledE.^2));
        for im = 1:numel(models)
            hereM = hereT & (Job.list.model == models(im));
            for N = Job.Nsweep(:).'
                d = Nmax/N;
                onGrid = hereM & all(mod(Job.list.ix, d) == 0, 2);
                M = zeros(1, Puse+1);
                for i2 = 1:3
                    sel = onGrid & (Job.list.X == i2);
                    M = M + (Job.weights{i1}(1, i2)*d^2)*sum(mu(sel, :), 1);   % |det C|/N^2 = (|det C|/Nmax^2)*d^2
                end
                row = table;
                row.t = i1;
                row.N = N;
                row.model = models(im);
                row.DoS = gather(((g.*M)*T).*ChebW);
                out = [out; row]; %#ok<AGROW>
            end
        end
    end
    Job.list = out;
end

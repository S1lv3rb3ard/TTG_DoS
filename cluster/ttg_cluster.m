function ttg_cluster(stage)
% TTG_CLUSTER  Cluster driver for the TTG Chebyshev-KPM DoS pipeline.
%
%   ttg_cluster('make')     Build the job list + Hamiltonians   -> out/job.mat        (CPU)
%   ttg_cluster('slice')    Compute this SLURM array task's moment slice
%                           -> out/partials/partial_XXXX.mat                           (1 GPU)
%   ttg_cluster('collect')  Gather all partials into one Job     -> out/job_full.mat   (CPU)
%
% getDoS is NOT run here: 'collect' just writes out/job_full.mat (the full Job struct
% with every per-slice moment row). Run getDoS on it locally to reconstruct the DoS.
%
% The three stages are normally launched (and auto-chained) by cluster/submit_all.sh.
% Everything lands in cluster/out/ next to this file, regardless of where MATLAB
% was started from.

    here = fileparts(mfilename('fullpath'));   % .../ttg/cluster
    repo = fileparts(here);                    % .../ttg
    addpath(genpath(repo));                    % put the whole codebase on the path
    outdir = getenv('TTG_OUTDIR');             % set by submit_all.sh (or your own shell)
    if isempty(outdir); outdir = fullfile(here, 'out'); end   % default: <repo>/cluster/out
    if ~exist(outdir, 'dir'); mkdir(outdir); end

    switch lower(string(stage))
        case "make",    makeStage(outdir);
        case "slice",   sliceStage(outdir);
        case "collect", collectStage(outdir);
        otherwise
            error('ttg_cluster:badStage', ...
                'Unknown stage "%s" (use make | slice | collect).', stage);
    end
end

% ======================================================================
%  Stage 1: build the job list (this is the top of test.m, saved to disk)
% ======================================================================
function makeStage(outdir)
    % ============ EDIT THESE TO MATCH YOUR RUN (same as test.m) ============
    theta = [-1.4, 0, 2.8
             -sqrt(3)/2, 0, sqrt(2)/2];
    NumStepW = 4;
    NumStepL = 25;
    N = 3:3:30;
    P = 4e3;
    % ======================================================================

    [W1, W2] = ndgrid(0:NumStepW);
    vecW = [W1(:) W2(:)]';
    vecW(:, 1) = [];
    JobList = table;

    NumTheta = size(theta, 1);
    L = zeros(NumTheta, 1);
    W = zeros(NumTheta, 1);
    X = cell(NumTheta, 1);
    H = cell(NumTheta, 1);
    scale = zeros(NumTheta, 1);
    weights = cell(NumTheta, 1);
    for i1 = 1:size(theta, 1)
        stack = getStack(1.42*sqrt(3), theta(i1, :));

        minW = getMinW(stack);
        stepW = [norm(stack.B{1}(:, 1)-stack.B{2}(:, 1)), norm(stack.B{2}(:, 1)-stack.B{3}(:, 1))];
        validW = stepW*vecW;
        validW = validW(:, validW<0.85 & validW>minW);
        W(i1) = median(validW);

        stepL = max(vecnorm(stack.B{2}, 2, 1));
        L(i1) = NumStepL*stepL;

        DoF = getDoF(stack, W(i1), L(i1), 'clean');
        X{i1} = getMoireBasis(stack, DoF, W(i1));
        H{i1} = getHamiltonianTB(stack, DoF, 8, 6.25);
        H0 = H{i1}(stack.K(:, 2));
        maxBasis = max(cellfun(@(c) size(c, 2), X{i1}));
        m = size(H0, 2);
        cost = 24*nnz(H0)+24*m+3*16*m*maxBasis+8*(P+1);
        for i2 = 1:numel(N)
            [Q, weights{i1}(i2, :)] = getMoireBZ(stack, N(i2));
            weights{i1}(i2, :) = weights{i1}(i2, :)/(6*stack.V^2);
            for i3 = 1:3
                numQ = size(Q{i3}, 2);
                temp = table;
                temp.t = repmat(i1, numQ, 1);
                temp.X = repmat(i3, numQ, 1);
                temp.N = repmat(i2, numQ, 1);
                temp.Q = Q{i3}.';
                temp.cost = repmat(cost, numQ, 1);
                JobList = [JobList; temp];
            end
        end
        scale(i1) = 1./(1.25*norm(H0, inf));
    end
    JobList.model = repmat("TB", size(JobList, 1), 1);

    Job.list = JobList;
    Job.theta = theta;
    Job.W = W;
    Job.L = L;
    Job.H = H;
    Job.X = X;
    Job.N = N;
    Job.weights = weights;
    Job.scale = scale;
    Job.P = P;

    save(fullfile(outdir, 'job.mat'), 'Job', '-v7.3');
    fprintf('make: wrote %s  (%d jobs)\n', fullfile(outdir, 'job.mat'), height(JobList));
end

% ======================================================================
%  Stage 2: one SLURM array task = one GPU = one strided slice of the jobs
% ======================================================================
function sliceStage(outdir)
    rank   = str2double(getenv('SLURM_ARRAY_TASK_ID'));     % 0-based task index
    nTasks = str2double(getenv('SLURM_ARRAY_TASK_COUNT'));  % total tasks in the array
    if isnan(nTasks)                                        % fallback for older Slurm
        nTasks = str2double(getenv('SLURM_ARRAY_TASK_MAX')) + 1;
    end
    assert(~isnan(rank) && ~isnan(nTasks), ...
        'ttg_cluster:noSlurm', ['Could not read SLURM_ARRAY_TASK_ID / _COUNT. ' ...
        'Run this stage from inside a Slurm array job.']);

    Job = load(fullfile(outdir, 'job.mat')).Job;
    Job.list = Job.list(rank+1 : nTasks : end, :);          % this task's share
    fprintf('slice: task %d of %d, %d jobs\n', rank, nTasks, height(Job.list));
    if height(Job.list) == 0
        fprintf('slice: empty slice, nothing to do.\n'); return
    end

    Job = BatchWeights(Job, "gpu");   % single visible GPU -> no pool (see getBatches guard)

    pdir = fullfile(outdir, 'partials');
    if ~exist(pdir, 'dir'); mkdir(pdir); end
    save(fullfile(pdir, sprintf('partial_%04d.mat', rank)), 'Job', '-v7.3');
    fprintf('slice: task %d done.\n', rank);
end

% ======================================================================
%  Stage 3: gather every per-slice partial into one full Job struct.
%  No cross-slice summation and no getDoS -- run getDoS locally on the result.
% ======================================================================
function collectStage(outdir)
    Job = load(fullfile(outdir, 'job.mat')).Job;   % full struct: H, X, weights, scale, P, ...

    files = dir(fullfile(outdir, 'partials', 'partial_*.mat'));
    assert(~isempty(files), 'ttg_cluster:noPartials', ...
        'No partial_*.mat files in %s', fullfile(outdir, 'partials'));
    parts = cell(numel(files), 1);
    for k = 1:numel(files)
        parts{k} = load(fullfile(files(k).folder, files(k).name)).Job.list;
    end

    % Keep every per-slice moment row as-is (no groupsummary). Each partial already
    % summed its own q-points per (t, N, X, model); all that's left is to add the
    % partials together, and getDoS does exactly that per (t, N, model) when you run
    % it locally. The DoS is linear in the moments, so the un-reduced rows give the
    % identical result -- this stage just collects them into one file.
    Job.list = vertcat(parts{:});

    save(fullfile(outdir, 'job_full.mat'), 'Job', '-v7.3');
    fprintf('collect: wrote %s  (full Job struct, %d moment rows from %d partials)\n', ...
        fullfile(outdir, 'job_full.mat'), height(Job.list), numel(files));
end

% ======================================================================
%  Helpers copied verbatim from test.m (so this file is self-contained)
% ======================================================================
function basis = getMoireBasis(stack, DoF, W)
invB = cellfun(@inv, stack.B, 'uni', false);
invB = blkdiag(invB{:});
compI = round(invB*DoF(:, 3:8).');
P = cat(2, stack.B{:});
compG = P*compI;

basis = cell(1, 3);
for j = 1:3
    if j == 1
        C = stack.B{1}-stack.B{2};
    elseif j == 2
        basis{2} = basis{1};
        continue;
    elseif j == 3
        C = stack.B{2}-stack.B{3};
    end
    limitN = floor(W*vecnorm(inv(C), 2, 2));
    ranN1 = -limitN(1):limitN(1);
    ranN2 = -limitN(2):limitN(2);
    [N1, N2] = ndgrid(ranN1, ranN2);
    N = [N1(:) N2(:)]';
    G = C*N;
    validG = G(:, vecnorm(G, 2, 1)<W);

    mask = pdist2(validG.', compG.')<1e-15;
    [~, c] = find(mask);

    idx = 2*(c-1)+(1:2);
    idx = idx.';
    idx = idx(:);

    mBasis = numel(idx);
    mDoF = 2*size(DoF, 1);

    basis{j} = sparse(idx, 1:mBasis, 1, mDoF, mBasis);
end
end

function [MoireBZ, weights] = getMoireBZ(stack, N)
    ranN = (0:N-1)/N;
    [X1, X2] = ndgrid(ranN);
    X = [X1(:) X2(:)].';
    MoireBZ = cell(1, 3);
    weights = zeros(1, 3);
    for j = 1:3
        if j == 1
            C = stack.B{1}-stack.B{2};
        elseif j == 2
            C = stack.B{1}-stack.B{2};
        elseif j == 3
            C = stack.B{2}-stack.B{3};
        end
        weights(j) = abs(det(C))/(N^2);
        MoirePoints = C*X;
        MoireBZ{j} = [MoirePoints+stack.K(:, j), MoirePoints+stack.pK(:, j)];
    end
end

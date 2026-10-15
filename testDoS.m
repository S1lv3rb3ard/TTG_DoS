theta = [-1.4, 0, 2.8];
        %-sqrt(3)/2, 0, sqrt(2)/2];
NumStepW = 4;
NumStepL = 5;   % quick test: start sites with |B_l n| >= L are missing (getMoireBasis warns); 8 suffices at W = 0.4324
N = 2;
P = 4e3;

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
    H{i1} = getHamiltonianTB(stack, DoF, 6.25, 0.25);   % interlayer cutoff = RelaxedTTG smoothCutoff on [6, 6.25] 1/A; all intralayer groups
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

clearvars -except Job
%%
tic
Job = BatchWeights(Job, "gpu");
toc

%%
%E = linspace(-0.2, 0.2, 4e3);
%Job = getDoS(Job, E, "gpu");
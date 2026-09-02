function Job = BatchWeights(Job, mode)
[BatchIdx, BatchCount, numGPUs, numCPUWorkers] = getBatches(Job.theta, Job.W, Job.L, Job.list, mode);

BatchList = cell(BatchCount, 1);
for i1 = 1:BatchCount
    idx = BatchIdx{i1};
    BatchList{i1} = Job.list(idx, :);
end

dq = parallel.pool.DataQueue;
afterEach(dq, @(idx) fprintf('Computed: batch %d of %d\n', idx, BatchCount));
fprintf('Starting DoS computation: %d batches total...\n', BatchCount)
%parfor (i1 = 1:BatchCount, numCPUWorkers)
for i1 = 1:BatchCount
   if strcmp(mode, 'gpu') && numGPUs>0
        t = getCurrentTask();
        if ~isempty(t)
            gpuId = mod(t.ID-1, numGPUs)+1;
            currDev = gpuDevice();
            if currDev.Index ~= gpuId
                gpuDevice(gpuId);
                fprintf('Worker %d successfully bound to GPU %d\n', t.ID, gpuId);
            end
        end
   end
   nJobs = size(BatchList{i1}, 1);
   [cellH, X1] = deal(cell(1, nJobs));
   hereH = Job.H; 
   hereX = Job.X; 
   hereScale = Job.scale;
   for i2 = 1:nJobs
       idxT = BatchList{i1}.t(i2);
       idxX = BatchList{i1}.X(i2);
       cellH{i2} = clean(hereScale(idxT).*hereH{idxT}(BatchList{i1}.Q(i2, :).'));
       %normest(cellH{i2})
       X1{i2} = hereX{idxT}{idxX};
   end
   mu = ChebyshevMoments(cellH, X1, Job.P, mode);
   BatchList{i1}.mu = gather(mu);
   send(dq, i1);
end

JobList = cell2mat(BatchList);
JobList.cost = [];
%JobList = groupsummary(JobList, ["t", "N", "X", "model"], "sum", "mu");
%JobList.GroupCount = [];
%%JobList = renamevars(JobList, 'sum_mu', 'mu');
Job.list = JobList;
end
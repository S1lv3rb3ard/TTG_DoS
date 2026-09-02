function [BatchIdx, BatchCount, numGPUs, numCPUWorkers] = getBatches(theta, W, L, JobList, mode)
localCluster = parcluster('local');
maxCPUWorkers = localCluster.NumWorkers;

memTol = 0.5;   % VRAM headroom: the 2-moments-per-matvec trick in ChebyshevMoments
                % allocates inner-product temporaries beyond the 3 vectors the cost
                % model reserves. 0.9 packed batches too tight and OOM'd on the A100.
mJobs = size(JobList, 1);
currentPool = gcp("nocreate");
if ~isempty(currentPool)
    numCPUWorkers = currentPool.NumWorkers;
else
    numCPUWorkers = 0;
end
numGPUs = gpuDeviceCount("available");
if strcmp(mode, 'gpu')
    assert(numGPUs>0, 'GPU mode requested but no GPUs are available.')
    workersPerGPU = 1;
    if numGPUs > 1
        % Multiple GPUs on this machine: one worker per GPU.
        desiredWorkers = numGPUs*workersPerGPU;
        dynamicWorkers = min(desiredWorkers, mJobs);
        if ~isempty(currentPool)
            if numCPUWorkers ~= dynamicWorkers
                delete(currentPool)
                currentPool = parpool('local', dynamicWorkers);
            end
        else
            currentPool = parpool('local', dynamicWorkers);
        end
        numCPUWorkers = currentPool.NumWorkers;
    else
        % Single visible GPU (e.g. one SLURM array task = one GPU):
        % run the batch loop serially on this device, no parpool.
        numCPUWorkers = 0;
    end

    minMem = inf;
    for i1 = 1:numGPUs
        minMem = min(minMem, (gpuDevice(i1).AvailableMemory));
    end
    memAvailable = (memTol*minMem)/workersPerGPU;
elseif strcmp(mode, 'cpu')
    numGPUs = 0;

    dynamicWorkers = min(maxCPUWorkers, mJobs);
    if ~isempty(currentPool)
        if numCPUWorkers~=dynamicWorkers
            delete(currentPool);
            currentPool = parpool('local', dynamicWorkers);
        end
    else
        currentPool = parpool('local', dynamicWorkers);
    end
    numCPUWorkers = currentPool.NumWorkers;
    memAvailable = (memTol*getAvailableMemory())/numCPUWorkers;
else
    error('Computation mode not supported: gpu or cpu only.')
end
BadJob = find(JobList.cost>memAvailable, 1);
if ~isempty(BadJob)
    if strcmp(mode, 'gpu')
        memType = 'GPU VRAM';
    else
        memType = 'CPU RAM (per worker)';
    end

    BadJobT = JobList(BadJob, 1);
    BadJobW = W;
    BadJobL = L(BadJobT);
    error('Job %d (W=%.2f, L=%d, Theta = %d) requires %.2f GB. This exceeds the available %s limit of %.2f GB!', ...
        BadJob, BadJobW, BadJobL, theta(BadJobT, :), JobList.cost(BadJob)/1e9, memType, memAvailable/1e9);
end
BatchIdx = cell(mJobs, 1);
BatchCount = 0;
BatchStart = 1;
currentMem = 0;
for i1 = 1:mJobs
    if (currentMem+JobList.cost(i1))>memAvailable
        BatchCount = BatchCount+1;
        BatchIdx{BatchCount} = BatchStart:(i1-1);
        BatchStart = i1;
        currentMem = JobList.cost(i1);
    else
        currentMem = currentMem+JobList.cost(i1);
    end
end
if BatchStart <= mJobs
    BatchCount = BatchCount+1;
    BatchIdx{BatchCount} = BatchStart:mJobs;
end
BatchIdx = BatchIdx(1:BatchCount);
end
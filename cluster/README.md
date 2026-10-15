# Running the TTG DoS on SuperMike-III

This folder runs your Chebyshev-KPM density-of-states sweep across many A100 GPUs.
The idea: your q-points are independent and only get summed at the end, so we **chop
the job list into 32 pieces, run them on separate GPUs at once, then gather the
results into one file.** Slurm (the cluster's scheduler) places the pieces on whatever
GPUs are free. (Adding the pieces up + the DoS reconstruction is done locally when you
run `getDoS` — see below.)

You only run **one command**. Everything else is automatic.

## What the pieces are

| File | What it is |
|------|------------|
| `ttg_cluster.m` | One MATLAB file with 3 stages: `make` (build job list), `slice` (one GPU does its share), `collect` (gather all slices into one `job_full.mat`). |
| `submit_all.sh` | The only thing you run. It hands the 3 stages to Slurm, chained so each waits for the previous. |
| `out/` | Created automatically. Holds `job.mat`, the `partials/`, the logs, and the final `job_full.mat`. |

Your whole `ttg` repo must be on the cluster — these scripts call `getStack`,
`getDoF`, `getHamiltonianTB`, `BatchWeights`, etc.

## One-time setup

**1. Get the code onto the cluster.** From your laptop, copy the whole `ttg` folder
to your home/work directory on SuperMike-III (e.g. `scp -r` or `rsync -a`, or `git clone`
if it's in a repo). Keep the folder structure intact.

**2. Find your two cluster-specific values** (log into SuperMike-III and run):
```bash
balance              # -> your allocation/account name (the thing that pays for it)
module avail matlab  # -> the available MATLAB module name, e.g. matlab/r2023b
```

**3. Fill them into `submit_all.sh`.** Either edit the two lines near the top by hand,
or run this once (replace the two values):
```bash
cd ~/ttg/cluster      # wherever you put the repo
sed -i 's/REPLACE_ALLOCATION/your_alloc_here/g; s|REPLACE_MATLAB_MODULE|matlab/r2023b|g' submit_all.sh
```

## Run it

From the repo root:
```bash
bash cluster/submit_all.sh
```
That prints three job IDs. Watch them with:
```bash
squeue -u $USER
```
When everything finishes, the cluster produces **`cluster/out/job_full.mat`** — the full
`Job` struct with every per-slice moment row. Copy it back and reconstruct the DoS
locally (`getDoS` is intentionally not run on the cluster):
```matlab
S = load('cluster/out/job_full.mat');   % S.Job.list has columns t, N, X, ix, model, mu
E = linspace(-0.2, 0.2, 4e3);           % your energy grid
Job = getDoS(S.Job, E, "cpu");          % or "gpu"; one DoS row per (theta, N in Nsweep)
```

## Knobs you might touch

- **Physics parameters** (`theta`, `Nmax`, `Nsweep`, `P`, step counts): top of the
  `makeStage` function in `ttg_cluster.m`. Moments are computed once on the `Nmax` grid;
  `getDoS` then returns the DoS for every `N` in `Nsweep` (each must divide `Nmax`, since an
  N-grid is a subset of the Nmax-grid). Only the K valley is computed; K' enters through
  the doubled weight in `getMoireBZ` (exact time-reversal symmetry of the truncated model).
- **Smaller P from the same run**: `getDoS(Job, E, mode, Puse)` uses moments 0..Puse, which
  are exactly the moments of a Puse run, so a P-convergence check needs only the largest P.
- **Energy grid** for the final DoS: set `E` when you call `getDoS` locally (see "Run it").
- **Number of slices / concurrency**: `NSLICES` and `MAXCONC` at the top of
  `submit_all.sh`. The 72 h wall limit is **per slice**, not for the whole run — the
  array can churn through tasks for days, only each individual slice must finish in 72 h.
  So make `NSLICES` big enough that `(total q-points / NSLICES) x time-per-q-point` stays
  comfortably under 72 h (the `make` log prints the total q-point count; smoke-test a few
  to get the per-q-point time). `MAXCONC` only sets how many slices run at once, i.e. the
  total wall-clock; raise it if your allocation allows more GPUs.
- **Output location**: `OUTDIR` at the top of `submit_all.sh` (default `cluster/out`). It
  holds `job.mat`, the `partials/`, the logs, and `job_full.mat`. On HPC your home quota
  is small, so send it to scratch/work:
  `OUTDIR=/work/$USER/ttg_out bash cluster/submit_all.sh` (use an absolute path; mind any
  scratch purge policy and copy `job_full.mat` back promptly).

## If something goes wrong

- **`$'\r': command not found` or `bad interpreter`** → the shell script picked up
  Windows line endings on the way over. Fix once on the cluster:
  `sed -i 's/\r$//' cluster/submit_all.sh` (or `dos2unix cluster/submit_all.sh`).
- **Smoke-test first.** Before a full run, set `NSLICES=4` in `submit_all.sh` to confirm
  the pipeline runs end to end cheaply, then put it back to 32.
- **`Unrecognized function 'renamevars'`** → your MATLAB module is too old (needs
  R2020a+). `module avail matlab` and pick a newer one.
- **License errors when many slices start at once** → lower `MAXCONC`, or ask me for the
  MATLAB Compiler recipe (compiles to a license-free executable).
- **"Job … exceeds the available GPU VRAM"** → a single q-point is too big for one A100;
  reduce the basis size (`W`) or the lattice (`NumStepL`) in `makeStage`.
- **A slice times out (72 h)** → that's the per-slice limit, not a total one. When a slice
  dies, `collect` won't start (it waits on the whole array via `afterok`). Raise `NSLICES`
  so each slice carries fewer q-points, then resubmit the slice + collect stages — `make`'s
  `job.mat` is still valid, so you don't rebuild it. (Changing `NSLICES` re-splits the
  q-points, so old `partials/` from a different `NSLICES` can't be reused — clear them
  first.) If slices are so large that even many of them risk 72 h, ask me to add
  checkpointing so a slice resumes across multiple windows instead.
- **Check a slice's log**: `cluster/out/logs/slice_<index>.out`.

## How the math stays correct

Each slice computes Chebyshev moments for its own q-points and sums them per
`(theta, N, layer)` group (your `BatchWeights` already does this — keeping the layer
separate so per-layer weights can be applied later). `collect` just stacks those
per-slice partial sums into one table without combining them; `getDoS` (run locally)
adds them up per `(theta, N)` when it reconstructs the DoS. Because the moments sum
associatively and the DoS is linear in them, splitting the q-points across 32 GPUs
gives exactly the same answer as running them all in one place.

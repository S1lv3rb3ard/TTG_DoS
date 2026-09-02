#!/bin/bash
# =====================================================================
#  TTG Chebyshev-KPM DoS  --  SuperMike-III submission driver
#
#  Usage (from anywhere):   bash cluster/submit_all.sh
#
#  Queues three auto-chained stages:
#     1. make    (CPU, single)  build the job list  -> cluster/out/job.mat
#     2. slices  (32x 1 A100)   compute moments      -> cluster/out/partials/
#     3. collect (CPU, single)  gather partials      -> cluster/out/job_full.mat
#  Each stage starts only if the previous one succeeded. getDoS is NOT run here --
#  load cluster/out/job_full.mat locally and run getDoS to reconstruct the DoS.
# =====================================================================
set -e

# ---- EDIT THESE TWO LINES --------------------------------------------
ALLOC="hpc_incomdos01"        # your account, from the `balance` command
MATMOD="matlab/r2025a"    # from `module avail matlab`, e.g. matlab/r2023b
# ----------------------------------------------------------------------

NSLICES=96        # GPU slices: keep each well under 72 h (32 gave ~69 h/slice -> 96 ~= 23 h)
MAXCONC=32        # slices at once = wall-clock lever (~2200 GPU-h / MAXCONC); Slurm caps to your GPU limit

# Where ALL output goes: job.mat, partials/, logs/, and the final job_full.mat.
# Default is cluster/out next to these scripts. On HPC your home quota is usually small
# and out/ holds 32 partials + the Hamiltonians, so point this at scratch/work instead:
#   OUTDIR=/work/$USER/ttg_out  bash cluster/submit_all.sh
OUTDIR="${OUTDIR:-cluster/out}"

cd "$(dirname "$0")/.."           # move to the repo root (parent of cluster/)
mkdir -p "${OUTDIR}/logs" "${OUTDIR}/partials"
OUTDIR="$(cd "${OUTDIR}" && pwd)" # absolutize so the MATLAB jobs (run from cluster/) agree
echo "Output -> ${OUTDIR}"

# 1) Build the job list + Hamiltonians (CPU).
# NOTE: this cluster's job_submit plugin prints SU-balance lines to stdout, which would
# pollute --parsable. Pipe through grep so we capture ONLY the numeric job id.
jid_make=$(sbatch --parsable <<EOF | grep -oE '^[0-9]+$' | tail -n1
#!/bin/bash
#SBATCH -J ttg-make
#SBATCH -A ${ALLOC}
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o ${OUTDIR}/logs/make.out
module load ${MATMOD}
export TTG_OUTDIR="${OUTDIR}"
cd "\${SLURM_SUBMIT_DIR}/cluster"
matlab -nodisplay -nosplash -batch "ttg_cluster('make')"
EOF
)
[ -n "${jid_make}" ] || { echo "ERROR: 'make' did not submit (no job id parsed from sbatch)."; exit 1; }
echo "[1/3] make   -> job ${jid_make}"

# 2) GPU array: one A100 per slice, starts after 'make' succeeds
jid_arr=$(sbatch --parsable --dependency=afterok:${jid_make} <<EOF | grep -oE '^[0-9]+$' | tail -n1
#!/bin/bash
#SBATCH -J ttg-dos
#SBATCH -A ${ALLOC}
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --array=0-$((NSLICES-1))%${MAXCONC}
#SBATCH -o ${OUTDIR}/logs/slice_%a.out
module load ${MATMOD}
export TTG_OUTDIR="${OUTDIR}"
cd "\${SLURM_SUBMIT_DIR}/cluster"
matlab -nodisplay -nosplash -batch "ttg_cluster('slice')"
EOF
)
[ -n "${jid_arr}" ] || { echo "ERROR: 'slices' did not submit (no job id parsed from sbatch)."; exit 1; }
echo "[2/3] slices -> job ${jid_arr}  (after ${jid_make})"

# 3) Gather all partials into one full Job struct (CPU), after the whole array succeeds
jid_red=$(sbatch --parsable --dependency=afterok:${jid_arr} <<EOF | grep -oE '^[0-9]+$' | tail -n1
#!/bin/bash
#SBATCH -J ttg-collect
#SBATCH -A ${ALLOC}
#SBATCH -p single
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 02:00:00
#SBATCH -o ${OUTDIR}/logs/collect.out
module load ${MATMOD}
export TTG_OUTDIR="${OUTDIR}"
cd "\${SLURM_SUBMIT_DIR}/cluster"
matlab -nodisplay -nosplash -batch "ttg_cluster('collect')"
EOF
)
[ -n "${jid_red}" ] || { echo "ERROR: 'collect' did not submit (no job id parsed from sbatch)."; exit 1; }
echo "[3/3] collect -> job ${jid_red}  (after ${jid_arr})"

echo
echo "Queued. Watch progress with:   squeue -u \$USER"
echo "Final result when finished:    ${OUTDIR}/job_full.mat  (then run getDoS locally)"

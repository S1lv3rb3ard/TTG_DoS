# TTG_DoS

MATLAB code for the regularized total density of states (DoS) and the momentum local DoS
of doubly incommensurate twisted trilayer graphene. It uses the momentum-space
(reciprocal) Hamiltonian of the ab initio tight-binding model, the W/L truncation scheme
and the kernel polynomial method (KPM), as described in Beard & Massatt, *Momentum Space
Algorithm for Electronic Structure of Double-Incommensurate Trilayer Graphene*.

Units: lengths in Å, momenta in Å⁻¹, energies in eV.

For changes made on 2026-10-07 (basis fix, time-reversal halving, nested N sweep,
multi-P reconstruction), see [`CHANGES.md`](CHANGES.md). For running on SuperMike-III,
see [`cluster/README.md`](cluster/README.md).

---

## Requirements

- MATLAB R2020a or later (`renamevars`, `groupsummary`).
- Statistics and Machine Learning Toolbox (`pdist2`).
- Parallel Computing Toolbox, for `gpuArray`, `parpool` and GPU mode.
- Optional: R2025b or later to run the KPM recursion in single precision
  (`ChebyshevMoments(..., 'single')`).
- Bundled in `3rd Party Addons/`: `ProgressBar` (provides `progress`) and matplotlib
  colormaps.

Put the whole repository on the path:

```matlab
addpath(genpath('path/to/TTG_DoS-main'))
```

---

## Quick start

`testDoS.m` is a small end-to-end run on one GPU: θ = (−1.4°, 0°, 2.8°), L = 5|b|,
N = 2, P = 4000. It builds the job list and computes the Chebyshev moments:

```matlab
testDoS                              % builds Job, runs BatchWeights(Job, "gpu")
E = linspace(-0.2, 0.2, 4e3);
Job = getDoS(Job, E, "gpu");         % Job.list.DoS: one row per (theta, N, model)
```

For production runs (several angle triples, N up to 30, L = 25|b|), use the cluster
pipeline, `bash cluster/submit_all.sh`, and then call `getDoS` locally on
`cluster/out/job_full.mat`.

---

## Pipeline

```
getStack ─► getDoF ─► getHamiltonianTB ─► getMoireBasis ─► getMoireBZ
 lattices   W/L-truncated   H(q) as a       KPM start       q-grids
            index set       function handle vectors         + weights
                                     │
                                     ▼
                BatchWeights ─► ChebyshevMoments ─► getDoS / getLDoS
                batches jobs     mu_n per q-point     Jackson-KPM reconstruction
                to fit memory    (GPU or CPU)         on an energy grid
```

1. **Geometry.** `getStack(aG, theta)` builds the three rotated layers. It returns the
   lattice and reciprocal bases `A{j}`, `B{j}`, the Dirac points `K`, `pK`, the B-site
   offsets `tau`, and `V = sqrt(|det B|)`. `getMinW` gives the smallest W for which the
   W-set is connected.
2. **Truncation.** `getDoF(stack, W, L, 'clean')` lists the reciprocal degrees of freedom
   (G, layer): sites in the L-ball whose folded momentum lies within W.
   - `'clean'` drops sites with no interlayer neighbour.
   - Each row carries both orbitals (A, B). Row r maps to matrix indices 2(r−1)+1 and
     2(r−1)+2.
3. **Hamiltonian.** `getHamiltonianTB(stack, DoF, tau, delta)` returns a handle
   `H(q)` that gives the sparse reciprocal Hamiltonian at momentum q.
   - Intralayer: every Jung–MacDonald (2013) Table II neighbour group, grouped by bond
     length (`getShells`); an optional `tauIntra` (Å) drops whole groups only.
   - Interlayer: Fourier-transformed Fang–Kaxiras hopping, multiplied by
     `getBump(|Q|, tau, delta)`. With `tau = 6.25`, `delta = 0.25` this is 1 up to 6 Å⁻¹ and
     falls smoothly to 0 at 6.25 Å⁻¹, identical to RelaxedTTG's cutoff.
   - `getHamiltonianBM` is the Bistritzer–MacDonald continuum model on the same index set.
     Its Dirac velocity defaults to the tight-binding value (`getMonolayerDirac`).
   - With zero relaxation, RelaxedTTG assembles the same H(q) to about 2e-7 of the largest
     element; the remainder is the spline interpolation of the interlayer transform.
4. **Start vectors.** `getMoireBasis(stack, DoF, W)` returns one set per q-grid X (next
   section).
5. **q-grids.** `getMoireBZ(stack, N)` returns an N × N grid on the moiré cell around K_X
   for each X, its weights, and integer grid labels.
6. **Moments.** `BatchWeights(Job, mode)` packs q-points into memory-sized batches
   (`getBatches`). It calls `ChebyshevMoments`, which runs the doubled KPM recursion
   (two moments per matvec) on a block-diagonal batch and stores
   μ_n = Σ_columns ⟨Y|T_n(sH)|Y⟩ for each q-point.
7. **Reconstruction.** `getDoS(Job, E, mode[, Puse])` applies the Jackson kernel
   (`Jackson`, `Chebyshev`) and the q-grid weights. `getLDoS` does the same for momentum
   LDoS line cuts (`getLinecut`, `getCenterBasis`), with weight 1/6 per centre orbital.

---

## How the total DoS is sampled

Each q-grid X samples one layer. By the shift identity (Eq. 4.55), the diagonal of
f(Ĥ_q) at a layer-l site with index G equals that layer's centre diagonal at the shifted
momentum q + ΣG (mod R*_l). The start vectors are the sites that turn the moiré-cell grid
into a uniform sampling of the W-disk:

| X | q-grid | moiré lattice C | layer sampled | start sites (G₁, G₂, G₃) |
|---|---|---|---|---|
| 1 | K₁ + C·[0,1)² | B₁ − B₂ | 3 | (B₁n, −B₂n, 0) |
| 2 | K₂ + C·[0,1)² | B₁ − B₂ | 2 | (B₁n, 0, 0) |
| 3 | K₃ + C·[0,1)² | B₂ − B₃ | 1 | (0, B₂n, −B₃n) |

Here n runs over all integer pairs with |C·n| < W, and n = 0 is the layer's centre site.
Layer 2 could equally use C = B₂ − B₃ with sites (0, 0, −B₃n).

**Valleys.** Only the K valley is computed. Its weight is doubled because the truncated
model is time-reversal symmetric: the DoF set is closed under G → −G and
H(−q) = Π·conj(H(q))·Π. The K′ contribution is therefore that of the grid −(K_X + C·x).

- For the tight-binding model this is an exact symmetry.
- For the BM model the doubling is the usual valley degeneracy.
- This relies on real hoppings and no magnetic field. It does not hold for driven
  Hamiltonians.

**Weights and normalization.** The weight of grid X is 2|det C|/N², divided by
6|Γ*| = 6V², where 6 = 3 layers × 2 orbitals. The DoS is normalized per state of the full
system, so it is the physical DoS inside the energy window. It does not integrate to 1,
because only the W-disks around the Dirac points are kept. When every start site exists
(L large enough), the integral over all energies is

    ∫ DoS dE = 2πW² / |Γ*|        (≈ 0.156 at W = 0.4324)

This is a quick check on L and on the number of start vectors. It cannot detect start
vectors placed on the wrong layer. For that, check that each grid's start sites lie on
the layer in the table above:

```matlab
[r, ~] = find(Job.X{t}{X});  unique(DoF(ceil(r/2), 9))   % should be the single layer listed
```

---

## The `Job` struct

| field | contents |
|---|---|
| `list` | table, one row per (theta, grid X, q-point): `t`, `X`, `N`, `ix` (grid labels), `Q` (momentum), `cost` (bytes), `model`, `mu` (moments 0..P) |
| `theta` | angle triples, one per row (degrees) |
| `W`, `L` | truncation radii per angle triple |
| `H` | cell of Hamiltonian handles `H{t}(q)` |
| `X` | cell of start-vector sets `X{t}{1:3}` |
| `weights` | `weights{t}(1, X)` = 2\|det C\|/(N²·6V²) |
| `scale` | Chebyshev scale s, the spectrum of s·H(q) must lie in (−1, 1); set to 1/(1.25‖H(K₂)‖∞) |
| `P` | number of Chebyshev moments computed |
| `Nmax`, `Nsweep` | grid size computed, and the N values `getDoS` reports (each divides `Nmax`) |
| `shift` | optional spectral centre (default 0): moments of s·(H − shift) |

---

## Main parameters

Set in `cluster/ttg_cluster.m` (`makeStage`) or at the top of `testDoS.m`.

| parameter | meaning | default |
|---|---|---|
| `theta` | twist angles of layers 1–3 (degrees) | [−1.4, 0, 2.8]; [−√3/2, 0, √2/2] |
| `NumStepW` | W is the median admissible combination of moiré steps below 0.85 Å⁻¹ | 4 |
| `NumStepL` | L = NumStepL·\|b\| | 25 (cluster), 5 (test) |
| `tau`, `delta` | interlayer momentum cutoff (Å⁻¹): flat to `tau − delta`, zero at `tau` | 6.25, 0.25 |
| `tauIntra` | intralayer bond-length cutoff (Å), whole groups only | Inf (all groups) |
| `Nmax`, `Nsweep` | q-grid computed, and the subgrids reported | 30, [3 5 6 10 15 30] |
| `P` | Chebyshev order; resolution ε ≈ π/(sP) | 4000 |

**Smaller P from one run.** `getDoS(Job, E, mode, Puse)` uses moments 0..Puse, which are
exactly those of a Puse-moment run. A P-convergence study therefore needs only the
largest P.

**Spectral scale.** The default bound uses ‖H‖∞ at K₂ with a 1.25 margin. ‖H(q)‖∞ varies
by about 20% over the q-grid. If the KPM moments grow instead of staying O(1), increase
the margin or use the maximum of ‖H(q)‖∞ over the grid.

---

## Checks worth running after changes

- **Shift identity.** For a start site G of grid X, the local moments at q should match
  the layer's centre moments at q + C·n to the truncation level (≈1e-8 for interior n).
- **Time reversal.** `H(-q)` should equal `Pi*conj(H(q))*Pi.'`, where `Pi` maps each
  DoF row to its G → −G partner. Compare the maximum absolute difference with machine
  precision.
- **Sum rule.** Σ_X weights · μ₀ equals the exact count Σ_X 2·w_X·#{n : |C_X n| < W}
  once L is large enough that every start site is in DoF (≈ 2πW²/|Γ*| up to lattice-point
  counting). `getMoireBasis` warns (`TTG_DoS:missingStartSites`) when sites are missing,
  e.g. at the `testDoS.m` value L = 5|b|.
- **`ttg_smoke.m`** runs these checks (except convergence) on the CPU in about a minute.
- **Convergence.** Compare the DoS across `Nsweep`, across `Puse`, and across W and L, as
  in Figs. 6–7 of the paper.
- **Monolayer.** `getMonolayerDirac(stack)` should give a Dirac energy of 3.17 meV and
  ħv_F = 5.577 eV·Å. A Dirac point away from K means partial neighbour groups.
- **Line cuts.** `getLinecut(K, Nq)` puts Γ_jk at the moiré-zone centre, the default as in
  RelaxedTTG. Pass `'legacy'` to reproduce the paths of the original Figs. 4–5.

These checks, and the 2026-10-07 changes, were validated against an independent NumPy
port of the tight-binding Hamiltonian, not in MATLAB. Smoke-test a run before relying on
it.

---

## Repository layout

```
Geometry/            getStack, getDoF, getMinW, getSparsity, getNumDoF,
                     getMoireBasis, getMoireBZ, getCenterBasis
Hamiltonians/
  Tight Binding/     getHamiltonianTB, intra-/interlayer operators, hopping tables,
                     getUnitaryTransform (maps to the BM gauge for comparisons)
  Bistritzer-MacDonald/  getHamiltonianBM and its operators
DoS/                 BatchWeights, getBatches, ChebyshevMoments, Chebyshev, Jackson,
                     getDoS, getLDoS, getLinecut, getAvailableMemory
cluster/             ttg_cluster.m (make / slice / collect), submit_all.sh, README.md
Auxiliary Functions/ rotd, rot, clean, kronDiv, setPlotting
3rd Party Addons/    ProgressBar, matplotlib colormaps
testDoS.m            small local run
CHANGES.md           change log with verification notes
```

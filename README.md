# UnrelaxedTTG.jl

Momentum-space tight-binding model of unrelaxed, doubly incommensurate twisted trilayer graphene:
the truncated reciprocal Hamiltonian Ĥ_q^{(τ,W,L)} and the spectral quantities computed from it
with the kernel polynomial method (momentum LDoS, density of states), on the CPU or with the
Chebyshev recurrence on a GPU.

Equation, lemma and section numbers in the code refer to the manuscript "Momentum-space algorithm
for the electronic structure of double-incommensurate twisted trilayer graphene" (K. Beard,
D. Massatt) in the version of 11 November 2026. The MATLAB code of this repository
(`TTG_DoS`, on the branch `main`) is the earlier implementation; `Model(stack; τ, δ, table = :II, shift = false)`
gives its intralayer model.

Units: lengths in Å, momenta in Å⁻¹, energies in eV, angles in degrees (counterclockwise).

## What the code computes

For a starting momentum q the labels are x = (G, tα): a layer t, a sublattice α ∈ {A, B} and
G = (G_s)_{s≠t} with G_s = B_s m_s, m_s ∈ ℤ². A label has the momentum k_x = q + ∑_{s≠t} G_s.

| Object | Formula | Code |
|---|---|---|
| Window (4.8), (4.27) | ‖ζ(x)‖₂ < W_t and max(‖G_k‖₂, ‖G_l‖₂) < L_t, with ζ(x) = q − K̃_t + ∑_{s≠t}(B_s − B_t) m_s (4.7) | `Window(stack, q, W, L)` |
| Intralayer block (2.4) | h̃_t(k_x) − E_D | `bloch` |
| Interlayer entry (3.5), (4.4) | c_j c_k g_τ(p) e^{i(G′_j·τ_{jα} − G_k·τ_{kβ})} ĥ_{kβ}^{jα}(p), p = k_x + G′_j | `neighbours`, `interlayer_entry`, `fourier_hopping` |
| Truncated Hamiltonian | Ĥ_q^{(τ,W,L)}, sparse and exactly Hermitian | `hamiltonian(model, window)`; `restrict` gives the one of a smaller (W, L) as a submatrix |
| Bound (4.35) | B_H ≥ ‖Ĥ_q^{(τ,W,L)}‖₂ | `spectral_bound(model)`, `spectral_bound(model, W)` |
| Chebyshev moments | μ_n = ∑_c [T_n(sĤ_q)]_cc, s = (1 − δ_s)/B_H | `chebyshev_moments` (CPU and GPU), `SlicedELL`, `reachable` |
| KPM density (4.33), (4.34) | s/(π√(1 − (sE)²)) ∑_n g_n^P μ_n T_n(sE) | `kpm_density`, `jackson` (kernel `Jackson()`, the default) |
| Delta-Chebyshev density (arXiv:2512.03149, (15)–(18)) | ∑_n ν_n(E) μ_n, the series of the rational kernel K_η of order m | `kpm_density(…; kernel = DeltaChebyshev(η, m))`, `truncation_bound` |
| Gaussian density (Section 2.5; KPM error note, Section 4) | ∑_{n<P} c_n(E) μ_n, the Chebyshev series of δ_ε(E − ·) | `kpm_density(…; kernel = Gaussian(ε))`, `truncation_bound` |
| Momentum LDoS (4.36) | (1/6) ∑_{jα} [δ^P_KPM(E, Ĥ_q)]_{(0,jα),(0,jα)} | `momentum_ldos`, `examples/production_linecut_gpu.jl` |
| Density of states (4.39), (4.1) | moiré-cell rule, Brillouin-zone rule | `moire_dos`, `brillouin_dos` |
| A posteriori diagnostics | Lemma 4.4, Remark 4.5, Lemma C.4 | `flux_bound`, `transfer_norm` (dense, small windows) |

The intralayer model is Jung–MacDonald (PRB 87, 195450) with the Dirac energy subtracted; the
interlayer model is by default the analytic refit of orders (6, 7) of Fang–Kaxiras (PRB 93, 235153),
with a closed-form Fourier transform, and as an option Fang–Kaxiras itself, Fourier transformed
numerically (F.8) (see "Choice of the interlayer model").

## Files

```
src/UnrelaxedTTG.jl     module, exports
src/geometry.jl         Stack: lattices A_j, B_j, Dirac points, nearest_dirac, connectedness_range
src/quadrature.jl       Gauss–Legendre, polar rule, momentum cutoff g_τ (4.4)
src/intralayer.jl       neighbour shells, Bloch matrix (2.4), Dirac energy, band gap (4.24)
src/interlayer.jl       hopping (F.5) with the models FANG_KAXIRAS, ANALYTIC_REFIT, ANALYTIC_REFIT_6_7, Hankel table and Fourier transform (F.8)
src/window.jl           Label, Window (direct enumeration), search_window (Theorem 4.3(b))
src/hamiltonian.jl      Model, hamiltonian, spectral_bound
src/parameters.jl       ANGLE_SETS, triple_parameters (angle triples with their W and L), claim_directory, moments_complete, write_moments, run_jobs
src/kpm.jl              kernels Jackson, DeltaChebyshev and Gaussian, kpm_density, truncation_bound, chebyshev_moments (CPU), gpu_matrix (stub)
src/sell.jl             reachable (rows the recurrence can reach), SlicedELL (storage for the GPU recurrence)
src/dos.jl              Linecut, momentum_ldos, moire_dos, brillouin_dos
src/posteriori.jl       a posteriori diagnostics of the truncation
src/reference.jl        slow reference implementations used by the tests
ext/UnrelaxedTTGCUDAExt.jl   gpu_matrix and chebyshev_moments on the GPU: kernel on SlicedELL, and CUSPARSE (loaded with CUDA.jl)
examples/               drivers, see below
test/runtests.jl        7377 tests; the GPU test set runs when CUDA.jl is installed and a GPU is found
```

## Installation

Julia 1.10 or later. The package is not registered. It is the branch `julia` of the repository
S1lv3rb3ard/TTG_DoS; install it from there, or use a local copy by path:

```julia
using Pkg
Pkg.activate("run")                          # any environment
Pkg.add(url = "https://github.com/S1lv3rb3ard/TTG_DoS", rev = "julia")
# or, for a local copy: Pkg.develop(path = "path/to/UnrelaxedTTG.jl")
Pkg.add("CUDA")                              # only for the GPU path
Pkg.test("UnrelaxedTTG")                     # about 1 min; CPU tests only
```

`Pkg.test` runs in an environment without CUDA.jl, so it skips the GPU test set. To include it, run
the test file in an environment that has CUDA.jl, UnrelaxedTTG, StaticArrays and SpecialFunctions
(the last two as direct dependencies, `Pkg.add(["StaticArrays", "SpecialFunctions"])`):

```
julia --project=run --threads=4 path/to/UnrelaxedTTG.jl/test/runtests.jl
```

The GPU path was run with Julia 1.13.3 and CUDA.jl 6.5.1 on an NVIDIA RTX 3080 (10 GB).

## Use on the CPU

```julia
using UnrelaxedTTG

stack  = Stack(1.42√3, (-1.4, 0.0, 2.8))          # lattice constant (Å), layer angles (degrees)
model  = Model(stack; τ = 8.5, δ = 6.5)            # cutoff g_τ of the interlayer hopping (Å⁻¹)
q      = stack.K[2]
window = Window(stack, q, 0.4742, 19.40)           # W, L (Å⁻¹)
H      = hamiltonian(model, window)                # sparse Ĥ_q^{(τ,W,L)}; label s ↦ rows 2s − 1 (A), 2s (B)

E      = range(-0.2, 0.2; length = 401)
B_H, _ = spectral_bound(model, 0.4742)
result = momentum_ldos(model, linecut(stack, 60), E; W = 0.4742, L = 19.40, P = 2000, B_H)
result.ldos                                        # momenta × energies, 1/eV
```

Start Julia with several threads (`julia -t auto`); the momenta are distributed over them.

## Choice of the kernel: Jackson, delta-Chebyshev or Gaussian

The Chebyshev moments μ_n do not depend on how the density is regularised. `kpm_density`,
`momentum_ldos`, `moire_dos` and `brillouin_dos` take the keyword `kernel`:

- `kernel = Jackson()` (default): the kernel polynomial method of the manuscript, (4.33)–(4.36).
  Positive; the resolution ε_P = π/(sP) is set by the number of moments.
- `kernel = DeltaChebyshev(η, m)`: the high-order regularized delta-Chebyshev method of Yi, Massatt,
  Horning, Luskin, Pixley and Kaye (arXiv:2512.03149). The δ-function is replaced by the rational
  kernel K_η(E, x) = −(1/π) ∑_l Im[w_l/(E − x + η z_l)] of order m with the equispaced poles
  z_l = 2l/(m + 1) − 1 + i of Colbrook, Horning and Townsend (SIAM Rev. 63 (2021) 489), and K_η is
  expanded in Chebyshev polynomials; the coefficients are used in closed form (docstring of
  `kpm_density`). η (eV) is the smoothing parameter. At a point where the density is m times
  continuously differentiable the error of the regularisation is O(η^m log(1/η)) (Theorem 1 of the
  reference), against O(P⁻²) for Jackson. For m ≥ 3 the kernel has negative lobes (−4.9 % of its
  peak for m = 6), so the density can be negative next to a sharp peak.

```julia
# continuing the example of "Use on the CPU" (model, E, B_H and result from there)
μ = result.μ[1, :]                                   # moments of the first momentum (or read from mu_0001.bin)
K = DeltaChebyshev(0.016, 6)                         # η = 16 meV, order 6
F = kpm_density(μ, result.s, E; kernel = K) ./ 6
b = truncation_bound(K, result.s, E, length(μ))      # bound for the terms n ≥ P left out, per unit μ₀ (1/eV)
result_K = momentum_ldos(model, linecut(stack, 60), E; W = 0.4742, L = 19.40, P = 2000, B_H, kernel = K)   # all momenta in one call
```

The series of K_η needs P of the order of 1/(sη) terms: the terms decay like exp(−n sη/√(1 − (sE)²)).
`truncation_bound` is a bound for the terms left out that holds for every set of moments
(|μ_n| ≤ μ₀); the difference to the density from fewer moments (`view(μ, 1:P′)`) measures them.
Widths: the full width at half maximum is 2.38 ε_P for the Jackson kernel and 0.66 η for m = 6
(2η, 1.46η, 1.13η, 0.92η, 0.77η for m = 1, …, 5), so Jackson with P moments and order 6 with
η = 11.2/(sP) have the same width.

Measured on 16 November 2026 from stored moments (set 2, L = 45.84, m = 6, energies in [−0.2, 0.2] eV,
s = 0.2321 eV⁻¹; `truncation_bound` and the largest difference of the LDoS between P = 4000 and all
8000 moments, 25 momenta, W = 0.3771, double precision):

| η | η sP at P = 4000 | `truncation_bound`, P = 4000 | measured, P = 4000 against 8000 |
|---|---|---|---|
| 8 meV | 7.4 | 12 eV⁻¹ | 8.4 × 10⁻² eV⁻¹ (1.9 × 10⁻² of the largest LDoS value) |
| 12 meV | 11.1 | 0.19 eV⁻¹ | 1.3 × 10⁻³ eV⁻¹ (4.4 × 10⁻⁴) |
| 16 meV | 14.9 | 3.5 × 10⁻³ eV⁻¹ | 5.6 × 10⁻⁶ eV⁻¹ (2.5 × 10⁻⁶) |
| 24 meV | 22.3 | 1.4 × 10⁻⁶ eV⁻¹ | 3.1 × 10⁻¹⁰ eV⁻¹ (1.0 × 10⁻¹⁰) |

- Single precision with `drop = 1e-5` against double precision with the full matrix changes the LDoS
  by the same amount for both kernels: 6.2 × 10⁻⁵ (Jackson) and 6.5 × 10⁻⁵ (η = 12 meV) in the
  measure (5.2), 34 momenta, W = 0.3771, P = 4000.
- For the momentum LDoS of this line cut the result still changes by about a third (measure (5.2))
  each time the resolution is halved, with either kernel (Jackson P = 1000 → 8000; η = 48 → 6 meV):
  its peaks are narrower than the resolutions reached, so the high order of the kernel does not show
  there. At equal width the two kernels differ by 7.8 × 10⁻² (η = 12 meV against Jackson P = 4000).
  The delta-Chebyshev LDoS of the line cut at W = 0.25 (η = 12 meV, P = 4000) is negative at 107 of
  301 301 points, down to −0.9 % of the largest value at the same momentum.
- Evaluating 1001 energies from 4000 moments takes 6 ms with the Jackson kernel and 45 ms with
  m = 6, per momentum on one core.

The order of convergence is tested on a density with known moments (the semicircle density, test
set "delta-Chebyshev kernel"): orders 1, 2, 4, 4, 6, 6 for m = 1, …, 6 in η, and 2 in 1/P for Jackson.
The total density of states, where the density is smooth, has not been run with this kernel.

### The Gaussian kernel (version 0.7.0)

- `kernel = Gaussian(ε)`: the Gaussian δ_ε(E − x) = exp(−(E − x)²/(2ε²))/(ε√(2π)) of Section 2.5,
  the regularisation in which the momentum LDoS and its error estimate are stated, expanded in
  Chebyshev polynomials: ρ(E) = ∑_{n<P} c_n(E) μ_n with δ_ε(E − x/s) = ∑_n c_n(E) T_n(x) on [−1, 1].
  The width ε is chosen independently of P. The terms left out are bounded by `truncation_bound`
  (Proposition 5 of the KPM error note of 16 November 2026), which decays like exp(−(Psε)²/2);
  P sε ≥ 4 makes it 1.6 × 10⁻⁴ of the peak 1/(ε√(2π)) or less. The Jackson kernel differs from
  δ_ε of the same nominal width by about 2.7 % of the peak, for every P. The truncated series can be
  slightly negative in gaps, by at most `truncation_bound`.

```julia
K = Gaussian(4 / (result.s * length(μ)))             # ε = 4/(sP)
F = kpm_density(μ, result.s, E; kernel = K) ./ 6
b = truncation_bound(K, result.s, E, length(μ))      # left-out terms and sampling, per unit μ₀ (1/eV)
```

The coefficients are sampled at the M = 2P Chebyshev nodes (a discrete cosine transform, written
as one sum over the nodes, with the sum over n done once per node), so their aliasing error is
that of the coefficients c_m, m > 3P, and is included in `truncation_bound`. Scripts take
`gauss=<ε in eV>` (`kernel_argument.jl`); the log line gives P sε.

Measured on 20 November 2026 with `examples/compare_kernels_linecut.jl` (801 energies in
[−0.2, 0.2] eV, 301 momenta, P = 4000; difference (3.4) per momentum, median over the momenta):

| line cut | comparison | median | worst |
|---|---|---|---|
| θ = (−√3/2°, 0°, √2/2°), W = 0.3771, L = 45.84 (F32, drop 10⁻⁵) | Gaussian ε = π/(sP) = 3.38 meV against Jackson | 1.4 × 10⁻² | 2.0 × 10⁻² |
| θ = (−1.39°, 0°, 2.4°), W = 0.35, L = 47.375 (F64, 8000 moments) | Jackson against the Gaussian LDoS at ε = π/(sP) = 3.22 meV | 2.5 × 10⁻² | 4.0 × 10⁻² |
| same | Gaussian kernel, ε = π/(sP), against the same | 1.5 × 10⁻³ | 2.3 × 10⁻³ |
| same | Gaussian kernel, ε = 4/(sP) = 4.10 meV, against the Gaussian LDoS at that ε | 5.4 × 10⁻⁵ | 1.0 × 10⁻⁴ |

The Gaussian LDoS of the second line cut is the Gaussian kernel with all 8000 moments; by
`truncation_bound` it is within 9.9 × 10⁻⁸ eV⁻¹ (ε = 3.22 meV) and 2.9 × 10⁻¹³ eV⁻¹ (ε = 4.10 meV) of
the Gaussian momentum LDoS (1/6) ∑_{jα} [δ_ε(E − Ĥ_q)]_{(0,jα),(0,jα)}. The Jackson peaks are higher
by about 1.7 %; the Gaussian series is negative at
2.8 % of the points of that line cut at ε = π/(sP), down to −0.0073 eV⁻¹ (0.07 % of its largest
value), and down to −1.3 × 10⁻⁴ eV⁻¹ at ε = 4/(sP). Evaluating the 301 momenta takes 0.2–0.3 s on
8 threads with either kernel.

## Choice of the interlayer model: Fang–Kaxiras or the analytic refit

The hopping between layers is h(x) = V₀(r) + V₃(r)Θ₃(x̂) + V₆(r)Θ₆(x̂), r = ‖x‖₂/a (F.5). Three radial
models are available; the angular factors Θ₃, Θ₆ are the same for all. The default
(`DEFAULT_INTERLAYER`) is the refit of orders (6, 7), the model adopted for the manuscript on
20 November 2026; until version 0.5 it was Fang–Kaxiras.

| name | constant | V₃, V₆ | Fourier transform |
|---|---|---|---|
| `:fang_kaxiras` | `FANG_KAXIRAS` | the functions (F.6) of Fang & Kaxiras | decays like ‖ξ‖⁻³ (not smooth at r = 0) |
| `:analytic_refit` | `ANALYTIC_REFIT` | r^m e^{−β_m r²/2} Σ_j v_{m,j} L_j^{(m)}(β_m r²), m = 3, 6, orders (2, 3) | decays like a Gaussian; closed form `closed_form_hankel` |
| `:analytic_refit_6_7` (default) | `ANALYTIC_REFIT_6_7` | the same form, orders (6, 7) | decays like a Gaussian, more slowly; closed form |

The refit keeps V₀ and replaces V₃ (3 coefficients, β₃ = 6.2595) and V₆ (4 coefficients,
β₆ = 3.9995) by their L²(r dr)-best approximations of this analytic form (note "A refit of the
Fang–Kaxiras interlayer hopping with a Gaussian-decaying Fourier transform", 17 November 2026). It
changes V₃ by at most 0.33 meV and V₆ by at most 0.061 meV, the coupling at the Dirac point from
104.01 to 103.95 meV, and the interlayer block of one pair of layers by about 0.9 meV in operator
norm. It is a fit to the published Fang–Kaxiras functions, not to DFT data.

`ANALYTIC_REFIT_6_7` is the refit of the same kind with 7 and 8 coefficients (β₃ = 8.6136,
β₆ = 5.3768). It changes V₃ by at most 0.158 meV and V₆ by at most 0.029 meV; on the production
windows the Hamiltonian changes by about 0.06 meV and the momentum LDoS by at most about 0.25 %
(report of 18 November 2026, Section 3). The price is a slower Gaussian decay of the transform,
exp(−ρ²a²/(2β₃)) with β₃ = 8.6136 instead of 6.2595. `AnalyticRefit` takes any number of
coefficients, and `linecut_gpu(…; interlayer = AnalyticRefit(…))` accepts a model object; its
meta.txt then holds the object with all parameters.

```julia
model  = Model(stack; τ = 8.5, δ = 6.5)                              # refit (6, 7), the default
FK     = Model(stack; τ = 8.5, δ = 6.5, interlayer = :fang_kaxiras)   # or parameters = FANG_KAXIRAS
model.interlayer                                                     # the radial model in use
```

The drivers use the default and take another model by name: `linecut_gpu(…; interlayer = :fang_kaxiras)`,
`choose_W(…; interlayer)`, `sweep_L(…; interlayer)`, and the argument `interlayer=fang_kaxiras` of
`production_linecut.jl`, `ldos_convergence.jl`, `ldos_linecut.jl` and `moire_dos.jl`
(`ldos_linecut.jl … legacy` keeps Fang–Kaxiras, the MATLAB model). A run with a refit writes the line
`interlayer <name>` (e.g. `interlayer analytic_refit_6_7`) into its meta.txt; a Fang–Kaxiras run writes
no such line, so its meta.txt is the same as before version 0.6.0 and older runs resume. To resume an
older Fang–Kaxiras run, pass `interlayer = :fang_kaxiras`; with the new default the meta.txt differs
and the driver stops instead of mixing models. Because
`claim_directory` compares meta.txt line by line, the moments of the two models cannot end up in one
folder. `ldos_from_moments.jl` needs no argument: the moments already belong to one model.
`examples/compare_interlayer.jl` compares the two models (transforms, Hamiltonian, LDoS of two line
cuts).

## Line cut on a GPU

```julia
using CUDA                                         # before the include
include("examples/production_linecut_gpu.jl")

linecut_gpu(; set = 2, W = 0.25, L = 45.84, P = 4000, Nq = 60, directory = "linecut_theta2",
              precision = ComplexF32, drop = 1e-5)
```

The Hamiltonians are assembled on the CPU threads and passed through a bounded queue to the GPU
tasks, which run the recurrence t_{m+1} = 2sĤ t_m − t_{m−1} for the six zero-label orbitals of
one momentum. Two moments are obtained per product (μ_{2m} = 2⟨t_m, t_m⟩ − μ₀,
μ_{2m+1} = 2⟨t_{m+1}, t_m⟩ − μ₁). See the docstring of `linecut_gpu` for all keyword arguments.

Two things make the GPU step fast (version 0.4.0):

- Only the rows that the recurrence can reach from the zero labels are kept (`reachable`): H̃ maps
  the span of these rows into itself, so the moments are those of the block on them. In the
  production windows about a third of the rows is not connected to the zero labels by Ĥ_q^{(τ,W,L)}.
- The block is stored in sliced ELLPACK form (`SlicedELL`), in which the 32 threads of a GPU warp
  read neighbouring memory, and one kernel launch per step computes the product, the update and
  both inner products. The inner products stay on the GPU until the end of the recurrence, and
  their sums have a fixed order, so a run reproduces its moments bit for bit.

By hand:

```julia
S = SlicedELL(H̃, ComplexF32; reach = c)              # on the CPU; c = start orbitals (rows of H̃)
μ = chebyshev_moments(gpu_matrix(S), c, P)             # on the GPU
```

`gpu_matrix(H̃, ComplexF32)` uploads compressed-row storage instead and uses the CUSPARSE product;
this is the path of versions 0.2.0 and 0.3.0, kept as an independent implementation for the tests.

Output, one folder per run:

- `meta.txt`: `key value…` lines (theta, W, L, P, Nq, B_H, J, delta_s, s, precision, drop_eV, tau,
  delta, table, interlayer (not for Fang–Kaxiras), varsigma (only for ς > 0), ticks, arclength).
- `mu_0001.bin`, …: the `P` moments of each momentum as raw `Float64`.

A run that is stopped continues from the files that exist and hold all `P` moments
(`moments_complete`; a file that is cut off or all zeros is computed again). Each file is written
to a temporary file, flushed to the disk and then renamed (`write_moments`), so a power failure
leaves no partial file. While a run is going, its folder is locked (`directory/.lock`,
`lock_directory`), so a second run started in the same folder is refused; a lock left behind by a
process that no longer runs on the same computer is taken over. A folder that holds moments is accepted
only if its `meta.txt` equals that of the present call (`claim_directory`); otherwise the call
throws an `ArgumentError` that names the differing parameters and writes nothing. If a task fails
(GPU out of memory, ‖Ĥ_q‖_∞ > B_H, …), the error is written to the log, the other tasks stop after
their current momentum, and the call throws; the same call then resumes the run. An interrupt
(Ctrl-C) is handled the same way: the call waits until every task has finished its momentum, so no
task writes into the folder after the call has returned.

The angle triples are those of `ANGLE_SETS`; `triple_parameters(set)` returns θ and the default W
and L, and throws for an unknown `set`.

The LDoS on an energy grid, for any P′ ≤ P:

```julia
μ = Vector{Float64}(undef, P); read!("linecut_theta2/mu_0001.bin", μ)
F = kpm_density(view(μ, 1:P′), s, E) ./ 6          # s from meta.txt, at full precision
```

`examples/ldos_from_moments.jl` does this for a whole folder; with the arguments `eta=0.016 m=6` it
uses the delta-Chebyshev kernel (see "Choice of the kernel").

### Choosing W and L

`examples/choose_W.jl` (it can be included on its own, after `using CUDA`) measures, at a few momenta of
the path, the relative difference (5.2) of the LDoS between candidate radii W and a reference
W_ref, all at the same L, P, arithmetic and KPM scale, together with the difference between the
production arithmetic and double precision with the full matrix (the floor). `choose_W` returns
the smallest W whose difference, and that of every larger candidate, is at most the floor;
`sweep_L` does the same comparison in L. No tolerance is prescribed.

Only the reference matrix is assembled at each momentum. For (W, L) ≤ (W_ref, L) the window is a
subset of the reference window (the conditions ‖ζ(x)‖ < W_t and label size < L_t only get stricter,
and K̃_t depends on q alone), and every entry depends only on q and its two labels. So the matrix of
a candidate is the principal submatrix of the reference on the labels that satisfy the smaller
bounds, `restrict(stack, window, H_ref, W, L)`; it is equal to `hamiltonian(model, Window(stack, q, W, L))`
entry for entry (tested). `sweep_L` does the same with (W, L_ref). For the next `ahead` momenta
(default 3) the reference matrix, its submatrices and their GPU format are prepared on CPU threads
while the GPU runs the recurrence of the current momentum.

```julia
using CUDA
include("examples/choose_W.jl"); include("examples/production_linecut_gpu.jl")
result = choose_W(; set = 2, L = 45.84, P = 4000, candidates = (0.20, 0.25, 0.30), W_ref = 0.3771)
linecut_gpu(; set = 2, W = result.choice, L = 45.84, P = 4000, Nq = 60, B_H = result.B_H,
              precision = ComplexF32, drop = 1e-5, directory = "linecut_theta2")
```

Pass `B_H` at full precision: the scale s enters the Jackson kernel, and a B_H rounded to four
decimals changes the LDoS by several 10⁻⁶ on its own. To resume a run, pass the `B_H` of its
`meta.txt` with all digits; a rounded value is refused as another parameter.

### Measured timings and accuracy

RTX 3080 and i5-13600KF (20 threads), θ = (−√3/2°, 0°, √2/2°), L = 45.84, P = 4000
(ε_P = 3.4 meV), GPU time of the recurrence per momentum; CUSPARSE path on 15 and 16 November
2026, `SlicedELL` on 16 November 2026:

| W | labels | arithmetic | CUSPARSE (0.2.0, 0.3.0) | `SlicedELL` (0.4.0) |
|---|---|---|---|---|
| 0.3771 | 336 000 | `ComplexF64`, full matrix | 30 s | 9.8 s |
| 0.3771 | 336 000 | `ComplexF32`, full matrix | 5.6 s | 1.9 s |
| 0.3771 | 336 000 | `ComplexF32`, `drop = 1e-5` | 3.0 s | 1.2 s |
| 0.25 | 156 000 | `ComplexF64`, full matrix | 12.7 s | 4.4 s |
| 0.25 | 156 000 | `ComplexF32`, full matrix | 2.0 s | 0.75 s |
| 0.25 | 156 000 | `ComplexF32`, `drop = 1e-5` | 1.4 s | 0.48 s |

The line cut of 301 momenta at W = 0.25 with `ComplexF32` and `drop = 1e-5` takes 146 s with
version 0.4.0 (417 s with version 0.2.0); its moments differ from those of the earlier run by at
most 3.0 × 10⁻⁶ and the LDoS by at most 5.0 × 10⁻⁷ in the measure (5.2). Building the storage on
the CPU takes 0.05–0.8 s per matrix (in the assembly tasks).

Relative LDoS differences (5.2) on [−0.2, 0.2] eV to double precision with the full matrix:
2.5 × 10⁻⁷ for `ComplexF32`, 2.3–2.6 × 10⁻⁵ with `drop = 1e-5` (the threshold sets the difference).
Assembly of one matrix with 156 000 labels takes 1.6 s on one core (version 0.3.0; 2.3–2.4 s for the
same matrix with version 0.2.0, 16 November 2026).

With `SlicedELL` a recomputed momentum reproduces the bits of the earlier file. With the CUSPARSE
path in `ComplexF32` the moments of the same matrix differ between two runs by about 10⁻⁶ (measured:
1.4 × 10⁻⁶ at P = 4000), because the default CUSPARSE product does not fix the order of its sums;
files written by versions 0.2.0 and 0.3.0 are therefore reproduced to about 10⁻⁶ only.

On 16 November 2026 the GPU moments were compared with the CPU ones on a window of 11 403 labels
for P ≤ 401: largest moment difference 1.2 × 10⁻¹⁴ in `ComplexF64` and 1.6 × 10⁻⁶ in `ComplexF32`.

## Other examples

| Script | Purpose |
|---|---|
| `ldos_linecut.jl` | momentum LDoS along the path on the CPU, written as a table |
| `production_linecut.jl` | CPU line cut with the same files as the GPU driver |
| `ldos_convergence.jl` | W- and L-convergence of the LDoS at ten momenta (CPU) |
| `ldos_from_moments.jl` | LDoS table from a folder of moments, with either kernel |
| `compare_kernels_linecut.jl` | Jackson against the Gaussian kernel on the stored moments of a line cut, with references from more moments if there are |
| `kernel_argument.jl` | the arguments `eta=<η>` and `m=<order>` (delta-Chebyshev kernel) and `gauss=<ε>` (Gaussian kernel) of `ldos_linecut.jl`, `ldos_from_moments.jl`, `moire_dos.jl`, and `interlayer=<name>` of the CPU scripts |
| `moire_dos.jl` | density of states by the moiré-cell rule (4.39) |
| `posteriori.jl` | a posteriori diagnostics of Lemma 4.4 |
| `cross_check_relaxed.jl` | comparison with `RelaxedTTG.jl` at zero relaxation |
| `compare_interlayer.jl` | Fang–Kaxiras against the analytic refit: transforms, ‖ΔĤ_q‖, LDoS of two line cuts |

## Conventions

- Lattice bases A_j have the columns a₁, a₂; reciprocal bases satisfy B_jᵀA_j = 2πI. The A
  sublattice of layer j is at 0 and the B sublattice at τ_j = A_j (1/3, 1/3).
- K_j = B_j (2/3, 1/3), K′_j = B_j (1/3, 2/3); K̃_t is the Dirac point of layer t nearest to q.
- Layers 1–2 and 2–3 are coupled; the coupling of the outer layers is neglected.
- A zero label that is not in the window contributes zero to the LDoS (convention (4.10)), so the
  average in (4.36) is always over six orbitals.
- λ is reserved for eigenvalues; the Fang–Kaxiras amplitudes λ₀, λ₃, λ₆ of the manuscript are
  `t₀`, `t₃`, `t₆` in the code.
- The sublattice signs of the interlayer angular factors are σ_A = −1, σ_B = +1 (the convention of
  `TTG_DoS` and `RelaxedTTG.jl`); see the docstring of `UnrelaxedTTG.angular` for the relation to
  (F.7).

## Known limitations

- `spectral_bound` estimates suprema on grids, so it is not a certified bound; every driver checks
  ‖Ĥ_q‖_∞ ≤ B_H for each matrix before the recurrence and stops if it fails (the momenta already
  running finish first; since 0.7.1 also in the CPU drivers).
- `drop > 0` and `ComplexF32` change the computed LDoS by the measured amounts above; neither is
  covered by the error estimates of the manuscript.
- The manuscript (Section 4.5 and its error estimates) is written for the Jackson kernel.
  `DeltaChebyshev` is an alternative evaluation of the same moments and is not covered by it.
  `Gaussian` evaluates the Gaussian regularisation of Section 2.5 itself, with the bound of
  Proposition 5 of the KPM error note (not yet in the manuscript).
- The W and L of sets 1 and 2 in `ANGLE_SETS` are the manuscript's values; (5.3)–(5.5) evaluated
  for a = 1.42√3 Å give 0.4748, 19.43 and 0.3777, 30.60 (fourth-digit difference).
- `claim_directory` compares `meta.txt` as text, so a folder is refused when a parameter is passed
  with other digits than those written there, even if the difference is rounding only (since
  0.7.2 the GPU driver writes W, L, B_H, drop and ς as `Float64`, so `drop = 0` and `0.0` agree).

# UnrelaxedTTG.jl

Momentum-space tight-binding model of unrelaxed, doubly incommensurate twisted trilayer graphene:
the truncated reciprocal Hamiltonian Ĥ_q^{(τ,W,L)} and the spectral quantities computed from it
with the kernel polynomial method (momentum LDoS, density of states), on the CPU or with the
Chebyshev recurrence on a GPU.

Equation, lemma and section numbers in the code refer to the manuscript "Momentum-space algorithm
for the electronic structure of double-incommensurate twisted trilayer graphene" (K. Beard,
D. Massatt) in the version of 11 November 2026. The MATLAB code in the rest of this repository
(`TTG_DoS`) is the earlier implementation; `Model(stack; τ, δ, table = :II, shift = false)`
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
| Truncated Hamiltonian | Ĥ_q^{(τ,W,L)}, sparse and exactly Hermitian | `hamiltonian(model, window)` |
| Bound (4.35) | B_H ≥ ‖Ĥ_q^{(τ,W,L)}‖₂ | `spectral_bound(model)`, `spectral_bound(model, W)` |
| Chebyshev moments | μ_n = ∑_c [T_n(sĤ_q)]_cc, s = (1 − δ_s)/B_H | `chebyshev_moments` (CPU and GPU) |
| KPM density (4.33), (4.34) | s/(π√(1 − (sE)²)) ∑_n g_n^P μ_n T_n(sE) | `kpm_density`, `jackson` |
| Momentum LDoS (4.36) | (1/6) ∑_{jα} [δ^P_KPM(E, Ĥ_q)]_{(0,jα),(0,jα)} | `momentum_ldos`, `examples/production_linecut_gpu.jl` |
| Density of states (4.39), (4.1) | moiré-cell rule, Brillouin-zone rule | `moire_dos`, `brillouin_dos` |
| A posteriori diagnostics | Lemma 4.4, Remark 4.5, Lemma C.4 | `flux_bound`, `transfer_norm` (dense, small windows) |

The intralayer model is Jung–MacDonald (PRB 87, 195450) with the Dirac energy subtracted; the
interlayer model is Fang–Kaxiras (PRB 93, 235153), Fourier transformed numerically (F.8).

## Files

```
src/UnrelaxedTTG.jl     module, exports
src/geometry.jl         Stack: lattices A_j, B_j, Dirac points, nearest_dirac, connectedness_range
src/quadrature.jl       Gauss–Legendre, polar rule, momentum cutoff g_τ (4.4)
src/intralayer.jl       neighbour shells, Bloch matrix (2.4), Dirac energy, band gap (4.24)
src/interlayer.jl       hopping (F.5), Hankel table and Fourier transform (F.8)
src/window.jl           Label, Window (direct enumeration), search_window (Theorem 4.3(b))
src/hamiltonian.jl      Model, hamiltonian, spectral_bound
src/kpm.jl              jackson, kpm_density, chebyshev_moments (CPU), gpu_matrix (stub)
src/dos.jl              Linecut, momentum_ldos, moire_dos, brillouin_dos
src/posteriori.jl       a posteriori diagnostics of the truncation
src/reference.jl        slow reference implementations used by the tests
ext/UnrelaxedTTGCUDAExt.jl   gpu_matrix and chebyshev_moments on CuSparseMatrixCSR (loaded with CUDA.jl)
examples/               drivers, see below
test/runtests.jl        6622 tests (CPU; the GPU extension is not covered)
```

## Installation

Julia 1.10 or later. The package is not registered; use it by path:

```julia
using Pkg
Pkg.activate("run")                          # any environment
Pkg.develop(path = "path/to/UnrelaxedTTG.jl")
Pkg.add("CUDA")                              # only for the GPU path
Pkg.test("UnrelaxedTTG")                     # about 40 s
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

## Line cut on a GPU

```julia
using CUDA                                         # before the include
include("examples/production_linecut_gpu.jl")

linecut_gpu(; set = 2, W = 0.25, L = 45.84, P = 4000, Nq = 60, directory = "linecut_theta2",
              precision = ComplexF32, drop = 1e-5)
```

The Hamiltonians are assembled on the CPU threads and passed through a bounded queue to the GPU
tasks, which run the recurrence t_{m+1} = 2sĤ t_m − t_{m−1} for the six zero-label orbitals of
one momentum as one sparse-times-dense product per step. Two moments are obtained per product
(μ_{2m} = 2⟨t_m, t_m⟩ − μ₀, μ_{2m+1} = 2⟨t_{m+1}, t_m⟩ − μ₁). See the docstring of `linecut_gpu`
for all keyword arguments.

Output, one folder per run:

- `meta.txt`: `key value…` lines (theta, W, L, P, Nq, B_H, J, delta_s, s, precision, drop_eV, tau,
  delta, table, ticks, arclength).
- `mu_0001.bin`, …: the `P` moments of each momentum as raw `Float64`.

A run that is stopped continues from the files that exist. Use a new folder when any parameter
changes: existing files are not checked against the arguments.

The LDoS on an energy grid, for any P′ ≤ P:

```julia
μ = Vector{Float64}(undef, P); read!("linecut_theta2/mu_0001.bin", μ)
F = kpm_density(view(μ, 1:P′), s, E) ./ 6          # s from meta.txt, at full precision
```

`examples/ldos_from_moments.jl` does this for a whole folder.

### Choosing W and L

`examples/choose_W.jl` (include it after `production_linecut_gpu.jl`) measures, at a few momenta of
the path, the relative difference (5.2) of the LDoS between candidate radii W and a reference
W_ref, all at the same L, P, arithmetic and KPM scale, together with the difference between the
production arithmetic and double precision with the full matrix (the floor). `choose_W` returns
the smallest W whose difference, and that of every larger candidate, is at most the floor;
`sweep_L` does the same comparison in L. No tolerance is prescribed.

```julia
include("examples/choose_W.jl")
result = choose_W(; set = 2, L = 45.84, P = 4000, candidates = (0.20, 0.25, 0.30), W_ref = 0.3771)
linecut_gpu(; set = 2, W = result.choice, L = 45.84, P = 4000, Nq = 60, B_H = result.B_H,
              precision = ComplexF32, drop = 1e-5, directory = "linecut_theta2")
```

Pass `B_H` at full precision: the scale s enters the Jackson kernel, and a B_H rounded to four
decimals changes the LDoS by several 10⁻⁶ on its own.

### Measured timings and accuracy

RTX 3080 and i5-13600KF (20 threads), θ = (−√3/2°, 0°, √2/2°), L = 45.84, P = 4000
(ε_P = 3.4 meV), per momentum, 15 November 2026:

| W | labels | arithmetic | GPU time |
|---|---|---|---|
| 0.3771 | 336 000 | `ComplexF64`, full matrix | 30 s |
| 0.3771 | 336 000 | `ComplexF32`, full matrix | 5.6 s |
| 0.3771 | 336 000 | `ComplexF32`, `drop = 1e-5` | 3.0 s |
| 0.25 | 156 000 | `ComplexF32`, `drop = 1e-5` | 1.3 s |

Relative LDoS differences (5.2) on [−0.2, 0.2] eV to double precision with the full matrix:
2.5 × 10⁻⁷ for `ComplexF32`, 2.3–2.6 × 10⁻⁵ with `drop = 1e-5` (the threshold sets the difference).
Assembly of one matrix with 156 000 labels takes 2.0 s on one core (16 November 2026).

On 16 November 2026 the GPU moments were compared with the CPU ones on a window of 11 403 labels
for P ≤ 401: largest moment difference 1.2 × 10⁻¹⁴ in `ComplexF64` and 1.6 × 10⁻⁶ in `ComplexF32`.

## Other examples

| Script | Purpose |
|---|---|
| `ldos_linecut.jl` | momentum LDoS along the path on the CPU, written as a table |
| `production_linecut.jl` | CPU line cut with the same files as the GPU driver |
| `ldos_convergence.jl` | W- and L-convergence of the LDoS at ten momenta (CPU) |
| `ldos_from_moments.jl` | LDoS table from a folder of moments |
| `moire_dos.jl` | density of states by the moiré-cell rule (4.39) |
| `posteriori.jl` | a posteriori diagnostics of Lemma 4.4 |
| `cross_check_relaxed.jl` | comparison with `RelaxedTTG.jl` at zero relaxation |

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
  ‖Ĥ_q‖_∞ ≤ B_H for each matrix before the recurrence and stops if it fails.
- `drop > 0` and `ComplexF32` change the computed LDoS by the measured amounts above; neither is
  covered by the error estimates of the manuscript.
- The scripts `production_linecut.jl` and `ldos_convergence.jl` know the angle sets 1–3 only (any
  other `set` is treated as set 3); `triple_parameters` in `production_linecut_gpu.jl` has set 4.
- If every GPU task of `linecut_gpu` fails (for example, out of GPU memory), the call does not
  return; check the log and the GPU memory before long runs.

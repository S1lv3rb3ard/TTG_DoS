# Changes

The changes were first checked against a NumPy port of the tight-binding Hamiltonian (not
included), at θ = (−1.4°, 0°, 2.8°), W = 0.4324 and L = 5|b|. On 2026-10-15 they were run in
MATLAB R2026a Update 5 with `ttg_smoke.m` (all checks pass; see section 0).

## 0. MATLAB run of 2026-10-15 (`ttg_smoke.m`) and a new warning in `getMoireBasis`

`ttg_smoke.m` (CPU, no parallel pool, about a minute) checks, at the parameters above:

| check | result |
|---|---|
| monolayer Dirac energy, ħv_F | 3.170 meV, 5.5767 eV·Å |
| H(q) Hermitian | 0 |
| DoF closed under G → −G (per layer) | 5075 of 5075 sites |
| H(−q) = Π conj(H(q)) Πᵀ | 9.0e-16 |
| K′ model at −q = conj(K model at q) | 6.2e-17 |
| start vectors: grid 1, 2, 3 on layer 3, 2, 1 only | yes |
| shift identity, layer 2, n = (1, 0), 24 moments | 5.9e-10 |
| sum rule Σ w μ₀ = exact site count, L = 8\|b\| | 0.145666 = 0.145666 |
| BM Dirac points of layers 1, 3 at K₁, K₃ | 0 |

The G → −G match must include the layer: a vector (0, 0, G₃) is the index of a layer-1 site
and of a layer-2 site.

**New warning (`TTG_DoS:missingStartSites`).** A start site (B₁n, −B₂n, 0) etc. is absent
from DoF when one of its components has |B_l n| ≥ L. `getMoireBasis` used to drop such sites
silently. At the `testDoS.m` setting L = 5|b| (W = 0.4324), 36 of 121 grid-1 sites and 32 of
121 grid-2 sites are missing, and Σ w μ₀ is 0.1186 instead of 0.1457, a 19% loss of mass
from holes in the tiling of the W-disk. At L = 8|b| all sites are present (5.57|b| is needed). The cluster
setting L = 25|b| covers every admissible W < 0.85 at both cluster angle sets (at most
23.3|b| is needed, at θ = (−√3/2°, 0°, √2/2°) and W = 0.85).
The exact reference for the sum rule is the site count, Σ_X 2·w_X·#{n : |C_X n| < W}; the
area estimate 2πW²/|Γ*| = 0.1559 differs from it by lattice-point counting.

## 1. Fix: layer 2 was sampled only at its centre (`Geometry/getMoireBasis.m`)

The old basis kept every site with G1+G2+G3 = C·n. By the shift identity, the diagonal at a
layer-l site with index G equals the layer-l centre at q + ΣG (mod R*_l), so this rule gives:

| grid (q-grid) | old basis | samples |
|---|---|---|
| X = 1 (C12, around K1) | layer-3 sites (B1 n, −B2 n, 0) + centres of layers 1, 2 | layer 3 (fine) + 2 extra centres |
| X = 2 (C12, around K2) | `basis{2} = basis{1}` | layer 3 again |
| X = 3 (C23, around K3) | layer-1 sites (0, B2 n, −B3 n) + centres of layers 2, 3 | layer 1 (fine) + 2 extra centres |

A layer-2 site has G2 = 0, so its total G1+G3 contains no B2. It never equals C12·n or C23·n
for n ≠ 0, so the rule cannot select layer-2 sites with either moiré lattice. As a result:
- layer 2 entered only through its centre site;
- layer 3 was counted twice;
- every grid also included all three centre sites.

The change is minimal. X = 1 and X = 3 keep their pairing and only drop the other layers'
centre sites. X = 2 gets the layer-2 sites (B1 n, 0, 0), which give the layer-2 centre at
q + B1 n ≡ q + C12 n. C23 with sites (0, 0, −B3 n) would work equally well, but the X = 2
q-grid is on C12.

Checks (local moments vs. the layer centre at q + C n, NumPy port):
- X = 1: 1.7e-8
- X = 2: 2.9e-8 (C23 alternative: 9.2e-9)
- X = 3: 3.5e-10

Effect at test size (L = 5|b|, N = 2, P = 1000, both valleys as before): the total DoS
changes by 53% (integrated |difference|).
- Most of the grid-3 change comes from the two dropped centre sites. They are the layer-2 and
  layer-3 centres at q near K3, where the near-flat bands sit, so each carries ~40x the weight
  of a typical start vector near E = 0.
- These centre terms each contribute one cell's worth of samples whatever N is, so the error
  does not shrink with N.

Sites are now matched component by component with tolerance 1e-9. The old 1e-15 test on a
recomputed total G missed 4 of the 65 grid-1 sites in the test case, because of rounding.

`cluster/ttg_cluster.m` had local copies of `getMoireBasis` and `getMoireBZ`. Local
functions shadow the path, so the copies are removed and the `Geometry/` versions are used.

## 2. Only the K valley is computed (`Geometry/getMoireBZ.m`): 2x

The truncated operator has an exact time-reversal symmetry:
- the DoF set is closed under G → −G;
- H(−q) = Π conj(H(q)) Π, checked to 2e-16;
- the summed moments at q and −q agree to 7e-16.

So K' is the time-reversed K grid −(K_j + C x), and only the weight is doubled. The old run
sampled K' at K'_j + C x instead. That is an equally valid representative, equal modulo R*_j.

The two differ by W-truncation error, because the fixed index window is not invariant under
q → q + R*_j. Measured on the total DoS (P = 200, N = 2), using a layer-j-on-grid-j basis.
The new basis is also closed under G → −G, so the symmetry argument is unchanged:

| W | 0.30 | 0.43 | 0.60 |
|---|---|---|---|
| max relative difference | 2.7e-3 | 1.9e-3 | 1.1e-3 |

The difference is unchanged with N (1.6–1.9e-3 for N = 2–4) and with L (1.90e-3 at L = 5|b|,
1.92e-3 at 8|b|). Expect old and new totals to differ at this level.

## 3. Nested N sweep (`ttg_cluster.m`, `getDoS.m`): 3.85x for the convergence study

An N-grid is a subset of the Nmax-grid whenever N divides Nmax: the points are the same
floating-point numbers, so the moments are the same. Moments are computed once at
Nmax = 30. `getDoS` returns the DoS for every N in `Nsweep = [3 5 6 10 15 30]`.

Cost: 900 grid points, against 3465 for N = 3:3:30. The N = 30 reference is unchanged.

## 4. Smaller P from one run (`getDoS(Job, E, mode, Puse)`)

Moments 0..Puse of a P-moment run are exactly the moments of a Puse run. So
P = 1000·2^(0:3) costs one P = 8000 run, against 15000 moments if run separately.

## Not changed

**Spectral scaling.** The true spectral half-width over sampled q is 4.14 eV, against the
code's 1.25·‖H(K2)‖∞ = 4.96 eV. That allows at most about 1.2x fewer moments, and it would
need a Lanczos bound at every q.

Note that ‖H(q)‖∞ ranges from 3.97 to 4.77 over q, compared with 3.97 at K2 alone. The 1.25
factor therefore has less slack than it appears. Bounding with the maximum of ‖H(q)‖∞ over
the grid (cheap) would be safer.

---

# 2026-10-08: consistency with RelaxedTTG

These changes make TTG_DoS, at zero relaxation, agree with RelaxedTTG. They were checked on
the NumPy port at θ = (−1.4°, 0°, 2.8°), W = 0.4324 and L = 5|b|.

## 5. Intralayer neighbour groups (`getShells`, `getHamiltonianTB`)

The old `getShells` cut A–B bonds at |R| < 8 Å instead of the bond length |R + τ_B|.
- **What it dropped.** A–A groups 6–7 entirely, and part of A–B groups 7–10 (2/3, 4/6, 2/6 and
  2/6 of the members).
- **Effect.** It broke C₃: the monolayer Dirac point sat 4.7e-4 Å⁻¹ off K, with a 5.1 meV
  splitting at K, a Dirac energy of 2.27 meV, and ħv_F of 5.515–5.577 eV·Å depending on
  direction.
- **Now.** Groups are formed by bond length and dropped only whole, and the default keeps all
  tabulated groups, as RelaxedTTG does. That gives a Dirac energy of 3.17 meV and an isotropic
  ħv_F = 5.577 eV·Å.
- **New parameter.** `getHamiltonianTB(stack, DoF, tau, delta, tauIntra)` separates the
  intralayer cutoff (Å) from the interlayer one (Å⁻¹). They previously shared `tau`.

## 6. Interlayer momentum cutoff (`testDoS.m`, `cluster/ttg_cluster.m`)

The drivers used `getBump(|Q|, 8, 6.25)`. That starts damping at 1.75 Å⁻¹, just above |K| = 1.70,
and damps the 2|K| couplings by 8%. They now use `getBump(|Q|, 6.25, 0.25)`, which is exactly
RelaxedTTG's `smoothCutoff(|Q|, 6, 6.25)`. It also keeps every evaluated |Q| inside the ±2π
interpolation grid, so the spline no longer extrapolates.

## 7. Line-cut Γ points (`getLinecut`)

Γ_jk is now the moiré-zone centre, at |K_k − K_j| from both corners; the old construction put
it at 1.118 |K_k − K_j|. `getLinecut(K, Nq, 'legacy')` reproduces the old paths. This matches
`rttg_common.getLinecut`.

## 8. Continuum model (`getHamiltonianBM`, `getIntralayerOperatorBM`, `getInterlayerOperatorBM`)

The construction labels each layer by the stored offset K_j − K_2 and selects hops by those
offsets. That makes it a consistent model in reflected coordinates, whose momentum q_c
corresponds to the physical momentum 2K_2 − q_c.
- **The old behaviour.** It was evaluated at q_c = q, so the Dirac points of layers 1 and 3
  sat at 2K_2 − K_j. At q = K_1 the layer-1 centre state was at 438 meV instead of 0, and
  continuum line cuts were mirrored through K_2 relative to tight binding.
- **The fix.** It is now evaluated at q_c = 2K_2 − q. Only the intralayer term depends on q,
  so this places every Dirac point at K_j.
- **Why not change the intralayer formula instead.** Correcting the intralayer formula alone
  would break the hop selection: only 3.6k of its 11.4k hops then coincide with strong
  tight-binding couplings, and the low-energy levels move further from tight binding.
- **Velocity.** v_F now defaults to the tight-binding value (`getMonolayerDirac`, 5.577 eV·Å)
  instead of 6.582·0.8 = 5.266 eV·Å. w_AA and w_AB are now arguments; the defaults are
  still 0.07 and 0.11 eV.

## Cross-code check (zero relaxation)

RelaxedTTG's u = 0 Hamiltonian was reimplemented independently: all groups, polar
quadrature (80 × 128 Gauss–Legendre, radius 8a) and smoothCutoff[6, 6.25]. It was compared on
the same DoF set (the `'clean'` set).

| | old TTG_DoS | fixed TTG_DoS |
|---|---|---|
| max \|ΔH\| / max \|H\| (three q-points) | 2.6–3.2e-3 | 1.4–1.7e-7 |
| 16 levels nearest E = 0 at K_2, max \|ΔE\| | 1.0 meV | 0.0003 meV |

- **Interlayer part alone:** 5.7e-4 before, 2.9e-6 after.
- **Quadrature convergence:** RelaxedTTG's polar quadrature agrees with a 160 × 256,
  radius-10a rule to 1e-14.

**Continuum versus tight binding** (v_F and w = ĥ(|K|) = 0.104 eV matched): the maximum
difference over the 12 levels nearest E = 0, in meV.

| q | old (at q) | fixed (at 2K_2 − q) |
|---|---|---|
| K_1 | 7.9 | 4.3 |
| K_2 | 4.3 | 4.3 |
| K_3 | 13.1 | 3.8 |

The remaining few meV is the difference between the models themselves.

**Results to recompute:** every TB DoS and line cut made with the old shells or the old cutoff
(shifts of about 1 meV near E = 0 at this size), and every continuum line cut.

## 9. Tolerant spectator matching (`getSparsity`, `getInterlayerOperatorBM`)

`getSparsity` paired interlayer sites by exact floating-point equality of their spectator
vectors. `getDoF` computes the same lattice vector by two routes, B·x0 for one layer and B·y
for the other, which can differ in the last bit.

**Effect in the NumPy port** (W = 0.4324, L = 5|b|):
- **Lost pairs.** 2,647 genuine pairs with |Q| ≤ 6.25 Å⁻¹ (9.3%) were dropped. The
  mismatch is per site, so `'clean'` removed 482 coupled sites as if they were isolated:
  4,577 kept against 5,059 correctly.
- **Levels.** The 16 levels nearest E = 0 at K_2 moved by up to 7.4 meV.
- **Now.** Matching uses `uniquetol`/`ismembertol` with a 1e-9 Å⁻¹ tolerance. `'clean'`
  then agrees with `'full'` to 3e-14 meV, removing only truly isolated sites. The continuum
  hop matching tolerance is raised from 1e-15 to 1e-9.

Whether MATLAB's products round the same way is not verified. To check, compare
`size(getDoF(stack, W, L, 'clean'), 1)` before and after this change.

## Lattice constant

The model is invariant under a change of a. The hopping amplitudes are given per neighbour
group, Fang–Kaxiras depends only on r/a, and the interlayer prefactor scales out.

Test: a = 2.439 Å against 2.4595 Å, with momenta and cutoffs scaled by the ratio. The 16
levels nearest E = 0 agree to 2e-5 meV, the spline interpolation floor. ħv_F |K| = 9.4976 eV
in both cases.

So a only sets the units of momentum: ħv_F = 5.577 eV·Å at 2.4595 Å, 5.530 eV·Å at
2.439 Å. Truncation parameters given in Å⁻¹ (W, L, the cutoffs) are in those units.

Note: the default L = NumStepL·|b| lies exactly on a lattice-vector norm. Which boundary
sites are kept therefore depends on rounding: 5,059 against 5,025 sites at the two lattice
constants, without a measurable change in the levels here. A value off the lattice norms,
for example (NumStepL + 0.5)·|b|, avoids the ambiguity.

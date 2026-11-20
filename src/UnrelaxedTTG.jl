"""
    UnrelaxedTTG

Momentum-space tight-binding model of unrelaxed, doubly incommensurate twisted
trilayer graphene: the truncated reciprocal Hamiltonian Ĥ_q^{(τ,W,L)} and the
spectral quantities computed from it (momentum LDoS, DoS, a posteriori bound of
the W-truncation).

Equation, lemma and section numbers refer to the manuscript "Momentum-space
algorithm for the electronic structure of double-incommensurate twisted
trilayer graphene" in the version of 11 November 2026
(`manuscript_EST_iop_2026-11-11`), which uses the Fang–Kaxiras interlayer model; the numbers of
the formulas are the same for the default refit model, only the radial functions V₃, V₆ differ.

Units: lengths in Å, momenta in Å⁻¹, energies in eV, angles in degrees
(counterclockwise). Conventions, as in `RelaxedTTG.jl`: lattice bases A_j with
columns a₁, a₂; reciprocal bases B_j with B_jᵀA_j = 2π I; the B sublattice of
layer j at τ_j and the A sublattice at 0; `(k, l) = others(j)` are the two
other layers in increasing order; label s of a `Window` carries the orbitals
2s − 1 (A) and 2s (B).

Typical use:

    stack  = Stack(1.42√3, (-1.4, 0.0, 2.8))
    model  = Model(stack; τ = 8.5, δ = 6.5)
    q      = stack.K[2]
    window = Window(stack, q, 0.4742, 19.40)
    H      = hamiltonian(model, window)                  # sparse Ĥ_q^{(τ,W,L)}
    ldos   = momentum_ldos(model, [q], range(-0.2, 0.2; length = 401); W = 0.4742, L = 19.40, P = 2000)

On a GPU (after `using CUDA`, which loads the extension `UnrelaxedTTGCUDAExt`) the Chebyshev
recurrence of Section 4.5 runs on the device; the windows and the assembly stay on the CPU:

    B_H, _ = spectral_bound(model, 0.4742)                # bound for windows of radius W
    s      = (1 - 0.05) / B_H                             # KPM scale s = (1 − δ_s)/B_H
    H̃      = UnrelaxedTTG.scaled(H, s, B_H)               # sĤ, after the check ‖Ĥ‖_∞ ≤ B_H
    c      = [o for t in 1:3 for o in orbitals(zero_label(window, t))]   # the six zero-label orbitals
    S      = SlicedELL(H̃, ComplexF32; reach = c)          # the rows reachable from c, in GPU storage (or ComplexF64)
    μ      = chebyshev_moments(gpu_matrix(S), c, 2000)
    F      = kpm_density(μ, s, range(-0.2, 0.2; length = 401)) ./ 6      # momentum LDoS (4.36)

The interlayer hopping is, by default (`DEFAULT_INTERLAYER`, since version 0.6.0), the analytic
refit of orders (6, 7) of the Fang–Kaxiras threefold and sixfold terms, whose Fourier transform
decays like a Gaussian (the model adopted for the manuscript on 20 November 2026). Fang–Kaxiras
itself is `Model(stack; τ = 8.5, δ = 6.5, interlayer = :fang_kaxiras)`, the coarser refit of
orders (2, 3) `interlayer = :analytic_refit`.

The same moments can be evaluated with the high-order delta-Chebyshev kernel of arXiv:2512.03149
or with a Gaussian instead of the Jackson kernel (keyword `kernel` of `kpm_density`, `momentum_ldos`, `moire_dos`,
`brillouin_dos`; the default `Jackson()` is the method of the manuscript):

    K      = DeltaChebyshev(0.03, 6)                      # η = 30 meV ≈ 15/(sP), order m = 6
    F      = kpm_density(μ, s, range(-0.2, 0.2; length = 401); kernel = K) ./ 6
    b      = truncation_bound(K, s, range(-0.2, 0.2; length = 401), length(μ))   # are P moments enough for this η?

or with the Gaussian δ_ε of Section 2.5 itself, expanded in Chebyshev polynomials (the
regularisation of the error estimate; width ε independent of P, P sε ≥ 4 recommended):

    K      = Gaussian(4 / (s * length(μ)))                # ε = 4/(sP)
    F      = kpm_density(μ, s, range(-0.2, 0.2; length = 401); kernel = K) ./ 6

(here every zero label is in the window; in general keep only those with `zero_label(window, t) > 0`,
convention (4.10)). `examples/production_linecut_gpu.jl` does this for a whole line cut, and
`examples/choose_W.jl` measures the W- and L-differences. See README.md for the list of files,
the file formats and the measured timings.
"""
module UnrelaxedTTG

using LinearAlgebra
using Printf
using SparseArrays
using SpecialFunctions: besselj
using StaticArrays

"""2-vector and 2×2 matrix (positions, momenta, lattice bases)."""
const Vec2{T} = SVector{2,T}
const Mat2{T} = SMatrix{2,2,T,4}

include("geometry.jl")        # Stack: lattices, Dirac points
include("quadrature.jl")      # Gauss–Legendre, polar rule, momentum cutoff g_τ
include("intralayer.jl")      # Jung–MacDonald shells, Bloch matrix, Dirac energy
include("interlayer.jl")      # Fang–Kaxiras hopping, its analytic refit, and their Fourier transforms
include("window.jl")          # labels, W- and L-truncation, connectedness search
include("hamiltonian.jl")     # Model and the assembly of Ĥ_q^{(τ,W,L)}
include("reference.jl")       # slow reference implementations for tests
include("kpm.jl")             # Chebyshev moments; densities with the Jackson, delta-Chebyshev or Gaussian kernel
include("sell.jl")            # reachable rows and sliced ELLPACK storage for the GPU recurrence
include("parameters.jl")     # angle triples of Section 5, output folders of the drivers
include("dos.jl")             # momentum LDoS, Brillouin-zone and moiré-cell DoS
include("posteriori.jl")      # a posteriori bound of the W-truncation (Lemma 4.4)

export Vec2, Mat2
export Stack, rotation, others, nearest_dirac, dirac_separation, connectedness_range
export gauss_legendre, polar_quadrature, momentum_cutoff
export Shells, jung_macdonald, bloch, dirac_energy, intralayer_bound, band_gap
export InterlayerModel, InterlayerParameters, FANG_KAXIRAS, AnalyticRefit, ANALYTIC_REFIT, ANALYTIC_REFIT_6_7, INTERLAYER_MODELS, DEFAULT_INTERLAYER, interlayer_model
export interlayer_hopping, closed_form_hankel, HankelTable, hankel, fourier_hopping
export Label, Window, zero_label, shifted_momentum, momentum, dimension, orbitals, search_window
export Model, hamiltonian, interlayer_entry, neighbours, interlayer_row_sum, spectral_bound, restrict
export ANGLE_SETS, triple_parameters, claim_directory, moments_complete, write_moments, run_jobs
export reference_hamiltonian, reference_fourier_hopping, reference_bloch, gaussian_density, reference_moments
export jackson, kpm_density, kpm_density!, chebyshev_moments, kpm_resolution, gpu_matrix
export Jackson, DeltaChebyshev, Gaussian, truncation_bound
export SlicedELL, reachable
export Linecut, linecut, momentum_ldos, moire_cell, moire_dos, brillouin_dos
export Boundary, boundary, transfer_norm, flux_bound, escape_norm, posteriori_bound, bracket_bound

end # module

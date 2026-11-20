# Cross-check against RelaxedTTG.jl (branch julia-implementation of DITTG_Projects) at zero
# relaxation: the two codes must give the same matrix entries on their common labels.
# Run in an environment that has both packages:
#
#   julia --project=<env> examples/cross_check_relaxed.jl
#
# Settings that make the models identical: Jung–MacDonald Table II without the Dirac shift
# (the choices of RelaxedTTG.jl) and the cutoff τ = 6.25, δ = 0.25, which is χ(·; 6.00, 6.25).

import RelaxedTTG, UnrelaxedTTG
using LinearAlgebra, StaticArrays
const R, U = RelaxedTTG, UnrelaxedTTG

a, θ = 1.42 * √3, (-1.4, 0.0, 2.8)
relaxed, unrelaxed = R.Stack(a, θ), U.Stack(a, θ)
q = unrelaxed.K[2] + SVector(0.013, -0.021)
basis = R.Basis(relaxed, 0.22, 7.0)
prepared = R.RelaxedHamiltonian(relaxed, basis, R.Shells(relaxed), R.zero_relaxation(relaxed); q, verbose = false,
    interlayer = (nᵣ = 200, n_φ = 256))
H_relaxed = Matrix(prepared(q))
model = U.Model(unrelaxed; τ = 6.25, δ = 0.25, table = :II, shift = false, parameters = U.FANG_KAXIRAS)
window = U.Window(unrelaxed, q, 0.3, 7.0)
H = Matrix(U.hamiltonian(model, window))
pairs = [(s, window.index[U.Label(basis.layer[s], basis.n[s])]) for s in 1:length(basis)
         if haskey(window.index, U.Label(basis.layer[s], basis.n[s]))]
rows = reduce(vcat, [[2s - 1, 2s] for (s, _) in pairs])
columns = reduce(vcat, [[2t - 1, 2t] for (_, t) in pairs])
println(length(pairs), " common labels of ", length(basis), " (RelaxedTTG) and ", length(window), " (UnrelaxedTTG)")
println("‖H_relaxed‖_F = ", norm(H_relaxed[rows, rows]), ",  ‖H_relaxed − H‖_F = ", norm(H_relaxed[rows, rows] - H[columns, columns]))

# Production line cut with checkpoints (manuscript, Figures 5.1(b) and 5.2(b)).
#
#   julia -t auto --project=. examples/production_linecut.jl <set> <P> <Nq> <directory> [budget in seconds] [W] [L] [interlayer=<name>]
#
# The interlayer model is `DEFAULT_INTERLAYER` (the refit of orders (6, 7)); `interlayer=<name>` selects another
# (`UnrelaxedTTG.INTERLAYER_MODELS`). A refit run has the line `interlayer <name>` in its meta.txt, a
# Fang–Kaxiras run (`interlayer=fang_kaxiras`) none.
#
# <set> = 1, …, 4 is the angle triple of `UnrelaxedTTG.ANGLE_SETS` with its default W = (W₁ + W*)/2 and L by (5.5):
# 1: θ = (−1.4°, 0°, 2.8°), W = 0.4742, L = 19.40;   2: θ = (−√3/2°, 0°, √2/2°), W = 0.3771, L = 30.56;
# 3: θ = (−1.39°, 0°, 2.4°), W = 0.4596, L = 18.95;  4: θ = (−1.39°, 0°, 2.8°), W = 0.4744, L = 19.555.
# Two more arguments override W and L. A folder that holds moments of other parameters is refused.
# For every momentum of the path the P Chebyshev moments of the zero-label orbitals are written to
# <directory>/mu_<index>.bin (P Float64 numbers). Momenta whose file exists are skipped, so the run
# can be stopped and started again; it stops by itself when the time budget is used up, and at the first
# momentum that fails (the momenta already running finish, then the error is thrown).
# `ldos_from_moments.jl` turns the moments into the LDoS on an energy grid, for any P′ ≤ P.

using UnrelaxedTTG, Printf, LinearAlgebra
include(joinpath(@__DIR__, "kernel_argument.jl"))         # interlayer_argument!, interlayer_meta
BLAS.set_num_threads(1)                                   # the momenta are distributed over the Julia threads

interlayer = interlayer_argument!(ARGS)
set, P, Nq, directory = parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), ARGS[4]
budget = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : Inf
θ, W, L = triple_parameters(set)                         # throws for an unknown set
length(ARGS) == 6 && throw(ArgumentError("give W and L together (arguments 6 and 7)"))
length(ARGS) ≤ 7 || throw(ArgumentError("unexpected arguments $(ARGS[8:end])"))
length(ARGS) == 7 && ((W, L) = (parse(Float64, ARGS[6]), parse(Float64, ARGS[7])))      # optional override

stack = Stack(1.42 * √3, θ)
model = Model(stack; τ = 8.5, δ = 6.5, interlayer)
path = linecut(stack, Nq)
δ_s = 0.05
B_H, 𝔍 = spectral_bound(model, W)
s = (1 - δ_s) / B_H
meta = sprint() do io
    println(io, "theta ", join(θ, " ")); println(io, "W ", W); println(io, "L ", L); println(io, "P ", P)
    println(io, "Nq ", Nq); println(io, "B_H ", B_H); println(io, "J ", 𝔍); println(io, "delta_s ", δ_s); println(io, "s ", s)
    println(io, "tau 8.5"); println(io, "delta 6.5"); println(io, "table I, Dirac shift E_D ", model.E_D)
    interlayer_meta(io, interlayer)
    println(io, "ticks ", join(path.ticks, " ")); println(io, "arclength ", join(path.s, " "))
end
claim_directory(directory, meta)                         # throws if the folder holds moments of a run with other parameters
file(i) = joinpath(directory, @sprintf("mu_%04d.bin", i))
pending = [i for i in eachindex(path.q) if !moments_complete(file(i), P)]   # a cut-off file is recomputed
@printf("%d of %d momenta to do; B_H = %.4f eV, ε_P = %.3f meV, %d threads\n", length(pending), length(path.q), B_H,
    1e3 * kpm_resolution(s, P), Threads.nthreads())
start = time()
done = run_jobs(pending; budget, start) do i            # stops at the first failing momentum
    window = Window(stack, path.q[i], W, L)
    orbitals_present = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
    H̃ = UnrelaxedTTG.scaled(hamiltonian(model, window), s, B_H)
    μ = chebyshev_moments(H̃, orbitals_present, P)
    write_moments(file(i), μ)
    @printf("q %4d: %6d labels, %d zero labels, %7.0f s elapsed\n", i, length(window), length(orbitals_present) ÷ 2, time() - start)
    flush(stdout)
end
@printf("finished %d momenta in %.0f s; %d remain\n", done, time() - start, length(pending) - done)

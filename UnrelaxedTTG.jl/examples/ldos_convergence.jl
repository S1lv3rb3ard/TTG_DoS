# Convergence of the momentum LDoS in W and L at fixed energy resolution (manuscript, eq. (5.2)).
#
#   julia -t auto --project=. examples/ldos_convergence.jl <set> <directory> [budget in seconds] [configurations] [Wmax] [interlayer=<name>]
#
# The interlayer model is `DEFAULT_INTERLAYER` (the refit of orders (6, 7)); `interlayer=<name>` selects another
# (`UnrelaxedTTG.INTERLAYER_MODELS`, e.g. `interlayer=fang_kaxiras`).
#
# At ten momenta of the path (its vertices and the midpoints of its segments) the moments of the
# zero-label orbitals are computed for several (W, L), all with the same KPM scale s and the same P,
# so that differences are truncation errors only. Files: <directory>/mu_<config>_<point>.bin.
# Finished files are skipped. The last configuration is the reference.

using UnrelaxedTTG, Printf, LinearAlgebra
include(joinpath(@__DIR__, "kernel_argument.jl"))         # interlayer_argument!, interlayer_meta
BLAS.set_num_threads(1)                                   # the momenta are distributed over the Julia threads

interlayer = interlayer_argument!(ARGS)
set, directory = parse(Int, ARGS[1]), ARGS[2]
budget = length(ARGS) ≥ 3 ? parse(Float64, ARGS[3]) : Inf
θ, W₀, L₀ = triple_parameters(set)                       # throws for an unknown set
configurations = set != 2 ?
    [(W₀, L₀), (W₀, 1.5L₀), (0.60, L₀), (0.60, 1.5L₀), (0.70, 2L₀)] :
    [(W₀, L₀), (W₀, 1.25L₀), (0.45, L₀), (0.45, 1.25L₀)]

# optional: explicit configurations "W,L;W,L;…" (the last one is the reference) and the scale radius Wmax
if length(ARGS) ≥ 4
    configurations = [Tuple(parse.(Float64, split(c, ','))) for c in split(ARGS[4], ';')]
end

stack = Stack(1.42 * √3, θ)
model = Model(stack; τ = 8.5, δ = 6.5, interlayer)
path = linecut(stack, 2)                                   # vertices and segment midpoints
points = path.q[1:end-1]                                   # K₁, ·, K₂, ·, Γ₁₂, ·, Γ₂₃, ·, K₂, · (K₃ is the 11th)
push!(points, path.q[end]); deleteat!(points, 9)           # drop the second K₂
Wmax = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : maximum(first, configurations)
B_H, _ = spectral_bound(model, Wmax)
s = 0.95 / B_H
P = 2 * round(Int, 8000 * B_H / spectral_bound(model, W₀)[1] / 2)     # ε_P as in the production run
haskey(ENV, "LDOS_P") && (P = parse(Int, ENV["LDOS_P"]))                # optional: prescribe the Chebyshev order
meta = sprint() do io
    println(io, "theta ", join(θ, " ")); println(io, "P ", P); println(io, "s ", s); println(io, "B_H ", B_H)
    println(io, "configurations ", join(["$(c[1]),$(c[2])" for c in configurations], " "))
    println(io, "points ", join(["$(p[1]),$(p[2])" for p in points], " "))
    interlayer_meta(io, interlayer)
end
claim_directory(directory, meta)                         # throws if the folder holds moments of a run with other parameters
file(c, i) = joinpath(directory, @sprintf("mu_%d_%02d.bin", c, i))
# the reference (last) configuration is computed at the five vertices of the path only
vertices = (1, 3, 5, 7, 10)
jobs = [(c, i) for c in eachindex(configurations) for i in eachindex(points)
        if !isfile(file(c, i)) && (c < length(configurations) || i in vertices)]
sort!(jobs; by = job -> -first(configurations[job[1]])^2 * last(configurations[job[1]])^2)      # largest first
@printf("%d jobs; P = %d, B_H = %.4f eV, ε_P = %.3f meV\n", length(jobs), P, B_H, 1e3 * kpm_resolution(s, P))
start = time()
Threads.@threads :dynamic for (c, i) in jobs
    time() - start < budget || continue
    W, L = configurations[c]
    window = Window(stack, points[i], W, L)
    present = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
    length(present) == 6 || error("a zero label is missing at point $i for W = $W")
    H̃ = UnrelaxedTTG.scaled(hamiltonian(model, window), s, B_H)
    μ = chebyshev_moments(H̃, present, P)                  # summed over the six zero-label orbitals
    write(file(c, i) * ".tmp", μ); mv(file(c, i) * ".tmp", file(c, i); force = true)
    @printf("W = %.4f, L = %.2f, point %2d: %6d labels, %6.0f s elapsed\n", W, L, i, length(window), time() - start)
    flush(stdout)
end
println("done")

# Density of states by the moiré-cell rule (4.39) (manuscript, Proposition 4.10).
#
#   julia -t auto --project=. examples/moire_dos.jl [N] [P] [output file]
#
# The centre labels x_m of the rule must lie in ℒ*(L): L_j > (W_j + √3β_jk)/(2 sin(θ_jk/2)).
# `moire_dos` reports the centres it had to skip in `result.missing`.

using UnrelaxedTTG, Printf

N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 4
P = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 1000
output = length(ARGS) ≥ 3 ? ARGS[3] : "moire_dos.txt"

stack = Stack(1.42 * √3, (-1.4, 0.0, 2.8))
model = Model(stack; τ = 8.5, δ = 6.5)
W = 0.3112                                              # W₁ of Section 5 for these angles
L = maximum(1.05 * (W + √3 * dirac_separation(stack, j, moire_cell(stack, j)[2])) /
            (dirac_separation(stack, j, moire_cell(stack, j)[2]) / stack.Knorm) for j in 1:3)
E = range(-0.3, 0.3; length = 301)
B_H, _ = spectral_bound(model, W)
elapsed = @elapsed result = moire_dos(model, E; W, L, N, P, B_H, verbose = true)

@printf("W = %.4f, L = %.2f, N = %d, P = %d: ε_P = %.2f meV, %d centres, %d missing, largest window %d labels, %.1f s\n",
    W, L, N, P, 1e3 * result.resolution, sum(result.centres), sum(result.missing), result.labels, elapsed)
@printf("mass %.6f, mass of the disks %.6f\n", result.mass, result.disk_mass)
open(output, "w") do io
    println(io, "# E (eV), DoS (1/(eV Å²))")
    for i in eachindex(E)
        println(io, E[i], " ", result.ρ[i])
    end
end
println("wrote ", output)

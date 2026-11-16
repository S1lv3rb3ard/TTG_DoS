# Momentum LDoS along the path K₁–K₂–Γ₁₂–Γ₂₃–K₂–K₃ (manuscript, Figure 5.2(b)) at the
# parameters of Section 5: θ = (−1.4°, 0°, 2.8°), τ = 8.5, δ = 6.5, W = 0.4742, L = 19.40.
#
#   julia -t auto --project=. examples/ldos_linecut.jl [P] [Nq] [output file] [legacy]
#
# With the fourth argument `legacy` the path uses the older Γ points of
# `getLinecut(K, Nq, 'legacy')` and the intralayer model is that of TTG_DoS
# (Jung–MacDonald Table II, no Dirac shift).
#
# Plot the LDoS against the arc length s (first column), not against the point index: the
# segments have different lengths.
#
# Writes a text file: one header line, then one line per momentum with the arc
# length s (Å⁻¹) followed by the LDoS at every energy (1/eV).

using UnrelaxedTTG, Printf

P = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 8000
Nq = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 20
output = length(ARGS) ≥ 3 ? ARGS[3] : "ldos_linecut.txt"

legacy = length(ARGS) ≥ 4 && ARGS[4] == "legacy"

stack = Stack(1.42 * √3, (-1.4, 0.0, 2.8))
model = legacy ? Model(stack; τ = 8.5, δ = 6.5, table = :II, shift = false) : Model(stack; τ = 8.5, δ = 6.5)
W, L = 0.4742, 19.40
path = linecut(stack, Nq; Γ = legacy ? :legacy : :moire)
E = range(-0.5, 0.5; length = 501)
B_H, 𝔍 = spectral_bound(model, W)          # ‖Ĥ_q^{(τ,W,L)}‖ ≤ B_H for this W; eq. (4.35) is `spectral_bound(model)`
elapsed = @elapsed result = momentum_ldos(model, path, E; W, L, P, B_H)

@printf("B_H = %.4f eV, 𝔍 = %.4f eV, s = %.5f 1/eV, ε_P = %.3f meV\n", B_H, 𝔍, result.s, 1e3 * result.resolution)
@printf("%d momenta, %d–%d labels per window, %.1f s on %d threads\n", length(path.q), extrema(result.labels)...,
    elapsed, Threads.nthreads())
open(output, "w") do io
    println(io, "# E (eV): ", join(E, " "))
    println(io, "# ticks: ", join(path.ticks, " "), " labels: ", join(path.labels, " "))
    for i in eachindex(path.q)
        println(io, path.s[i], " ", join(view(result.ldos, i, :), " "))
    end
end
println("wrote ", output)

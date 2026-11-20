# Momentum LDoS along the path K₁–K₂–Γ₁₂–Γ₂₃–K₂–K₃ (manuscript, Figure 5.2(b)) at the
# parameters of Section 5: θ = (−1.4°, 0°, 2.8°), τ = 8.5, δ = 6.5, W = 0.4742, L = 19.40.
#
#   julia -t auto --project=. examples/ldos_linecut.jl [P] [Nq] [output file] [legacy] [eta=<η>] [m=<order>] [gauss=<ε>] [interlayer=<name>]
#
# The interlayer model is `DEFAULT_INTERLAYER` (the refit of orders (6, 7)); `interlayer=<name>` selects another
# (`UnrelaxedTTG.INTERLAYER_MODELS`, e.g. `interlayer=fang_kaxiras`).
#
# With `eta=<η>` (eV) the moments are evaluated with the delta-Chebyshev kernel of order m
# (default 6) of arXiv:2512.03149, with `gauss=<ε>` (eV) with the Gaussian δ_ε expanded in Chebyshev
# polynomials, instead of the Jackson kernel (`kernel_argument.jl`).
#
# With the argument `legacy` (anywhere; the positional arguments before it may be left out) the path uses the older Γ points of
# `getLinecut(K, Nq, 'legacy')` and the intralayer model is that of TTG_DoS
# (Jung–MacDonald Table II, no Dirac shift); its interlayer model is then Fang–Kaxiras unless
# `interlayer=<name>` is given.
#
# Plot the LDoS against the arc length s (first column), not against the point index: the
# segments have different lengths.
#
# Writes a text file: three header lines (energies; ticks; model, W, L, P and kernel), then one line per momentum with the arc
# length s (Å⁻¹) followed by the LDoS at every energy (1/eV).

using UnrelaxedTTG, Printf
include(joinpath(@__DIR__, "kernel_argument.jl"))

kernel = kernel_argument!(ARGS)
legacy = "legacy" in ARGS                                  # decided once, wherever the word stands
filter!(!=("legacy"), ARGS)
interlayer = interlayer_argument!(ARGS; default = legacy ? :fang_kaxiras : DEFAULT_INTERLAYER)   # legacy: the MATLAB model
length(ARGS) ≤ 3 || throw(ArgumentError("unexpected arguments $(ARGS[4:end]); see the usage line at the top of this file"))
P = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 8000
Nq = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 20
output = length(ARGS) ≥ 3 ? ARGS[3] : "ldos_linecut.txt"

stack = Stack(1.42 * √3, (-1.4, 0.0, 2.8))
model = legacy ? Model(stack; τ = 8.5, δ = 6.5, table = :II, shift = false, interlayer) : Model(stack; τ = 8.5, δ = 6.5, interlayer)
W, L = 0.4742, 19.40
path = linecut(stack, Nq; Γ = legacy ? :legacy : :moire)
E = range(-0.5, 0.5; length = 501)
B_H, 𝔍 = spectral_bound(model, W)          # ‖Ĥ_q^{(τ,W,L)}‖ ≤ B_H for this W; eq. (4.35) is `spectral_bound(model)`
elapsed = @elapsed result = momentum_ldos(model, path, E; W, L, P, B_H, kernel)

@printf("B_H = %.4f eV, 𝔍 = %.4f eV, s = %.5f 1/eV, %s\n", B_H, 𝔍, result.s, kernel_text(kernel, result.s, P))
kernel isa Jackson || @printf("terms beyond P = %d: at most %.1e per eV (truncation_bound)\n", P,
    maximum(truncation_bound(kernel, result.s, E, P)))
@printf("%d momenta, %d–%d labels per window, %.1f s on %d threads\n", length(path.q), extrema(result.labels)...,
    elapsed, Threads.nthreads())
open(output, "w") do io
    println(io, "# E (eV): ", join(E, " "))
    println(io, "# ticks: ", join(path.ticks, " "), " labels: ", join(path.labels, " "))
    println(io, "# model: theta ", join(stack.θ, " "), legacy ? " table II, no Dirac shift, legacy path" : " table I, Dirac shift",
        ", interlayer ", interlayer, "; W ", W, " L ", L, " P ", P, " B_H ", B_H, " s ", result.s, "; ", kernel_text(kernel, result.s, P))
    for i in eachindex(path.q)
        println(io, path.s[i], " ", join(view(result.ldos, i, :), " "))
    end
end
println("wrote ", output)

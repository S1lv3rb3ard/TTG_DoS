# A posteriori diagnostics of Lemma 4.4 and Remark 4.5 for the zero label of layer 2.
#
#   julia --project=. examples/posteriori.jl
#
# Dense linear algebra: keep the window below a few thousand labels.

using UnrelaxedTTG, Printf

stack = Stack(1.42 * √3, (-1.4, 0.0, 2.8))
model = Model(stack; τ = 8.5, δ = 6.5)
q = stack.K[2]
Σ, ε, η = (-0.1, 0.1), 0.05, 0.3
L = 8.0
for W in (0.20, 0.25, 0.30)
    window = Window(stack, q, W, L)
    outside = boundary(model, window; L)               # compare with the L-truncated operator
    orbital = orbitals(zero_label(window, 2))[1]
    bound, flux = flux_bound(model, window, orbital; Σ, ε, η, outside)
    𝔟, g₀ = transfer_norm(model, window, W / 2; Σ, ε, η, outside)
    @printf("W = %.2f: %4d labels, %4d outside; flux bound %.3e; 𝔟 = %.3e eV, barrier gap %.3f eV, bound (4.22) %.3e\n",
        W, length(window), length(outside.labels), bound, 𝔟, g₀, posteriori_bound(𝔟, ε, η))
end

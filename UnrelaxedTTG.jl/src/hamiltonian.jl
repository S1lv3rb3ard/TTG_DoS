# The truncated reciprocal Hamiltonian Ĥ_q^{(τ,W,L)} (manuscript, Definition 3.3
# and Sections 4.2–4.4).

"""
    Model(stack; τ, δ, table = :I, shift = true, ς = 0, parameters = ANALYTIC_REFIT_6_7, hankel = (;))
    Model(stack; τ, δ, interlayer = :fang_kaxiras, …)

Everything that does not depend on the starting momentum q:

    stack     the three layers
    shells    intralayer neighbour shells (Jung–MacDonald `table`, Appendix F)
    E_D       the energy subtracted from the on-site terms: the Dirac energy
              (Section 2.4) if `shift`, otherwise 0
    ĥ         Hankel table of the interlayer hopping on [0, τ] (eq. (F.8))
    τ, δ      momentum cutoff g_τ of the interlayer hopping (eq. (4.4)), 0 < δ < τ (Å⁻¹)
    ς         damping length (Å) of the regularization (2.13): ĥ is multiplied by
              e^{−ς²‖ξ‖²/2}; ς = 0 is the model of Section 5
    interlayer  the radial model of the interlayer hopping ([`InterlayerModel`](@ref)):
              `parameters`, or `interlayer_model(interlayer)` if the name `interlayer`
              is given (`:fang_kaxiras`, `:analytic_refit` or `:analytic_refit_6_7`, see
              `INTERLAYER_MODELS`). Only one of the two keywords may be given; without
              either the model is `DEFAULT_INTERLAYER` (`ANALYTIC_REFIT_6_7`).

`hankel` are keyword arguments of [`HankelTable`](@ref).
"""
struct Model{T<:Real,I<:InterlayerModel}
    stack::Stack{T}
    shells::Shells{T}
    E_D::T
    ĥ::HankelTable{T}
    τ::T
    δ::T
    ς::T
    interlayer::I
end

function Model(stack::Stack{T}; τ::Real, δ::Real, table::Symbol = :I, shift::Bool = true, ς::Real = 0,
        parameters::Union{InterlayerModel,Nothing} = nothing, interlayer::Union{Symbol,Nothing} = nothing,
        hankel = (;)) where {T}
    0 < δ < τ || throw(ArgumentError("the cutoff needs 0 < δ < τ"))
    parameters === nothing || interlayer === nothing ||
        throw(ArgumentError("give the interlayer model either by `parameters` or by name (`interlayer`), not both"))
    p = interlayer !== nothing ? interlayer_model(interlayer) : something(parameters, interlayer_model(DEFAULT_INTERLAYER))
    ς ≥ 0 || throw(ArgumentError("ς must be nonnegative"))
    shells = Shells(stack.a; table)
    E_D = shift ? dirac_energy(shells) : zero(T)
    return Model{T,typeof(p)}(stack, shells, E_D, HankelTable(p, stack.a, T(τ); hankel...), T(τ), T(δ), T(ς), p)
end

"""
    interlayer_entry(model, a, b, p) -> 2×2 SMatrix

The momentum-dependent factor of an interlayer entry of Ĥ^τ_q between layers
`a` (rows) and `b` (columns) at the momentum argument `p`:

    c_a c_b g_τ(p) e^{−ς²‖p‖²/2} ĥ_{bβ}^{aα}(p),      c_a c_b = |Γ*|,

entry (α, β), in eV (Definition 3.3 with the cutoff of Section 4.2). It is
zero for ‖p‖₂ ≥ τ.
"""
@inline function interlayer_entry(model::Model{T}, a::Integer, b::Integer, p::Vec2{T}) where {T}
    ρ = norm(p)
    g = momentum_cutoff(ρ, model.τ, model.δ)
    g > 0 || return zero(SMatrix{2,2,Complex{T},4})
    stack = model.stack
    damping = model.ς > 0 ? exp(-(model.ς * ρ)^2 / 2) : one(T)
    return (stack.Γ * g * damping) * fourier_hopping(model.ĥ, p, stack.τ̂[a], stack.τ̂[b])
end

"""
    candidates(g, model, q, x; upper = false)

Call `g(y, b, p, G′ₐ, φ_b)` for every label y that the interlayer part of Ĥ^τ_q can couple to
the label `x`, without evaluating the hopping. For x = (G, a) and a coupled layer b with
spectator layer l, these are the labels y = (G′, b) with G′_l = G_l and G′_a = B_a n for the
n ∈ ℤ² with

    ‖p‖₂ < τ,      p = k_x + G′_a = q + G_b + G_l + G′_a,

the momentum argument of ĥ in (3.5). `φ_b` = e^{−iG_b·τ_b} is the phase of the B sublattice of
layer b, which does not depend on n. With `upper = true` only the layers b > a are visited
(each coupled pair once).

The n are found in the square of half-width ⌈τ/σ_min(B_a)⌉ + 1 about the rounded point
−B_a⁻¹k_x, which contains every n with ‖B_a n + k_x‖₂ < τ.
"""
function candidates(g::G, model::Model{T}, q::Vec2{T}, x::Label; upper::Bool = false) where {G,T}
    stack = model.stack
    a = x.layer
    Bₐ = stack.B[a]
    kₓ = momentum(stack, q, x)
    centre = -(Bₐ \ kₓ)                                    # p = B_a (n − centre)
    n₀ = round.(Int, centre)
    reach = ceil(Int, model.τ / stack.σ) + 1
    for b in coupled(a)
        upper && b < a && continue
        G_b = stack.B[b] * x.m[b]
        φ_b = cis(-(G_b ⋅ stack.τ[b]))                     # e^{−iG_b·τ_{bβ}} for β = B (1 for A)
        for d₂ in -reach:reach, d₁ in -reach:reach
            n = n₀ + SVector(d₁, d₂)
            G′ₐ = Bₐ * n
            p = kₓ + G′ₐ
            norm(p) < model.τ || continue
            y = Label(b, setindex(setindex(x.m, n, a), zero(n), b))
            g(y, b, p, G′ₐ, φ_b)
        end
    end
    return nothing
end

"""
    coupling(model, a, b, p, G′ₐ, φ_b) -> 2×2 SMatrix

The interlayer block of (3.5) between a label x of layer `a` and the label y of layer `b`
reached with the momentum argument `p` = k_x + G′_a:

    [Ĥ^τ_q]_{(x,α),(y,β)} = c_a c_b g_τ(p) e^{i(G′_a·τ_{aα} − G_b·τ_{bβ})} ĥ_{bβ}^{aα}(p),

entry (α, β), with `φ_b` = e^{−iG_b·τ_b} (τ_{jA} = 0, τ_{jB} = τ_j). It is the zero matrix where
g_τ(p) underflows to 0, just inside ‖p‖₂ = τ.
"""
@inline function coupling(model::Model{T}, a::Integer, b::Integer, p::Vec2{T}, G′ₐ::Vec2{T}, φ_b::Complex{T}) where {T}
    block = interlayer_entry(model, a, b, p)
    iszero(block) && return block
    φₐ = cis(G′ₐ ⋅ model.stack.τ[a])                       # e^{iG′_a·τ_{aα}} for α = B (1 for A)
    phases = SMatrix{2,2,Complex{T},4}(1, φₐ, φ_b, φₐ * φ_b)
    return phases .* block
end

"""
    neighbours(f, model, q, x; upper = false)

Call `f(y, block)` for every label y coupled to the label `x` by a nonzero
interlayer entry of Ĥ^τ_q, where `block[α, β]` = [Ĥ^τ_q]_{(x,α),(y,β)}. With
`upper = true` only the labels y of a layer above the layer of x are visited
(each coupled pair once).

For x = (G, a) and a coupled layer b with spectator layer l, the coupled labels
are y = (G′, b) with G′_l = G_l and G′_a = B_a n for the n ∈ ℤ² with
‖p‖₂ < τ, p = k_x + B_a n, and by (3.5)

    [Ĥ^τ_q]_{(x,α),(y,β)} = c_a c_b g_τ(p) e^{i(G′_a·τ_{aα} − G_b·τ_{bβ})} ĥ_{bβ}^{aα}(p).

The labels come from [`candidates`](@ref) and the blocks from [`coupling`](@ref). Every candidate
is evaluated, whether or not y belongs to a window; [`hamiltonian`](@ref) looks y up first.
"""
function neighbours(f::F, model::Model{T}, q::Vec2{T}, x::Label; upper::Bool = false) where {F,T}
    a = x.layer
    candidates(model, q, x; upper) do y, b, p, G′ₐ, φ_b
        block = coupling(model, a, b, p, G′ₐ, φ_b)
        iszero(block) || f(y, block)
        return nothing
    end
    return nothing
end

"""
    hamiltonian(model, window) -> SparseMatrixCSC

The truncated reciprocal Hamiltonian Ĥ_q^{(τ,W,L)} on the labels of `window`
(q = `window.q`): the intralayer blocks h̃_t(k_x) − E_D on the diagonal and the
interlayer entries (3.5) between labels of the window. Label
s occupies rows and columns 2s − 1 (A) and 2s (B). Every interlayer block is
computed once, from its end in the lower layer, and its conjugate transpose is
stored at the mirrored position, so the matrix is exactly Hermitian.

For each label x the labels y of [`candidates`](@ref) are looked up in the window before their
block [`coupling`](@ref) is evaluated, so no hopping is computed for labels outside the window
(about 60 % of the candidates at W = 0.25, L = 45.84). The entries are those of
[`neighbours`](@ref) restricted to the window.
"""
function hamiltonian(model::Model{T}, window::Window{T}) where {T}
    stack, q = model.stack, window.q
    n = dimension(window)
    I, J, V = Int[], Int[], Complex{T}[]
    sizehint!(I, 64n); sizehint!(J, 64n); sizehint!(V, 64n)
    for (s, x) in enumerate(window.labels)
        h̃ = bloch(model.shells, stack, x.layer, momentum(stack, q, x); E_D = model.E_D)
        for β in 1:2, α in 1:2
            push!(I, 2s - 2 + α); push!(J, 2s - 2 + β); push!(V, h̃[α, β])
        end
        candidates(model, q, x; upper = true) do y, b, p, G′ₐ, φ_b
            s′ = get(window.index, y, 0)
            s′ == 0 && return                              # y is not in the window: its block is not needed
            block = coupling(model, x.layer, b, p, G′ₐ, φ_b)
            iszero(block) && return
            for β in 1:2, α in 1:2
                push!(I, 2s - 2 + α); push!(J, 2s′ - 2 + β); push!(V, block[α, β])
                push!(I, 2s′ - 2 + β); push!(J, 2s - 2 + α); push!(V, conj(block[α, β]))
            end
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    restrict(stack, window, H, W, L) -> (H′, present, labels)

The truncation Ĥ_q^{(τ,W,L)} for W ≤ `window.W` and L ≤ `window.L` (per layer) as a principal
submatrix of `H` = Ĥ_q of `window`, without assembling it again. This is exact: the labels of
the window (W, L) are those of `window` with ‖ζ(x)‖₂ < W_t and max(‖G_k‖₂, ‖G_l‖₂) < L_t
(eqs. (4.8), (4.27)), at the same Dirac points K̃_t (which depend on q only), so the smaller
window is a subset of the larger one; and every entry of Ĥ_q depends only on q and its two
labels, not on W or L. The labels keep their order, so `H′` equals `hamiltonian(model,
Window(stack, q, W, L))` entry for entry (tested). `present` holds the orbital indices of the
zero labels in the smaller window, `labels` its number of labels.
"""
function restrict(stack, window, H, W, L)
    size(H) == (2length(window), 2length(window)) ||
        throw(DimensionMismatch("H is $(size(H)), the window has $(2length(window)) orbitals"))
    W, L = triple(eltype(window.W), W), triple(eltype(window.L), L)
    all(W .≤ window.W) && all(L .≤ window.L) || throw(ArgumentError("(W, L) must not exceed the window's"))
    keep = [s for (s, x) in enumerate(window.labels)
            if norm(window.ζ[s]) < W[x.layer] && label_size(stack, x) < L[x.layer]]
    position = zeros(Int, length(window))
    position[keep] = eachindex(keep)
    rows = [o for s in keep for o in orbitals(s)]
    zero = [position[s] for s in (zero_label(window, t) for t in 1:3) if s > 0]   # zero labels in the smaller window
    present = [o for p in zero if p > 0 for o in orbitals(p)]
    return H[rows, rows], present, length(keep)
end

"""
    interlayer_row_sum(model, a, k) -> SVector{2}

Absolute row sums ∑_y |[Ĥ^τ_{q,inter}]_{(x,α),y}|, α = A, B, of the untruncated
interlayer part for a label x of layer `a` with momentum k_x = `k`. The supremum
over k and a is the constant 𝔍 of eq. (4.5).
"""
function interlayer_row_sum(model::Model{T}, a::Integer, k) where {T}
    stack = model.stack
    Bₐ = stack.B[a]
    k = Vec2{T}(k)
    n₀ = round.(Int, -(Bₐ \ k))
    reach = ceil(Int, model.τ / stack.σ) + 1
    total = zero(SVector{2,T})
    for b in coupled(a), d₂ in -reach:reach, d₁ in -reach:reach
        p = k + Bₐ * (n₀ + SVector(d₁, d₂))
        norm(p) < model.τ || continue
        block = interlayer_entry(model, a, b, p)
        total += SVector(abs(block[1, 1]) + abs(block[1, 2]), abs(block[2, 1]) + abs(block[2, 2]))
    end
    return total
end

"""
    spectral_bound(model; n = 24, margin = 0.02) -> (B_H, 𝔍)

The q-independent bound of eq. (4.35),

    B_H = sup_k ‖h̃_t(k)‖₂ + 𝔍 ≥ sup_q ‖Ĥ_q^{(τ,W,L)}‖₂,

with the first term replaced by its row-sum bound ([`intralayer_bound`](@ref))
and 𝔍 (eq. (4.5)) estimated as (1 + `margin`) times the largest
[`interlayer_row_sum`](@ref) over an `n` × `n` grid of each layer's reciprocal
cell. The row sums are continuous and ℛ*_a-periodic in k, but the grid maximum
is an estimate of the supremum, not a bound; the KPM drivers therefore check
‖Ĥ_q‖_∞ ≤ B_H for every matrix they use.
"""
function spectral_bound(model::Model{T}; n::Integer = 24, margin::Real = 0.02) where {T}
    𝔍 = zero(T)
    for a in 1:3, i₂ in 0:n-1, i₁ in 0:n-1
        k = model.stack.B[a] * Vec2{T}(i₁ / n, i₂ / n)
        𝔍 = max(𝔍, maximum(interlayer_row_sum(model, a, k)))
    end
    𝔍 *= 1 + margin
    return intralayer_bound(model.shells; E_D = model.E_D) + 𝔍, 𝔍
end

"""
    spectral_bound(model, W; n = 24, margin = 0.02, nᵣ = 16, n_φ = 48) -> (B_H, 𝔍)

A smaller bound for the truncated matrices of window radius `W` (a number or a
triple). Every label x of 𝒲*(q, W) has k_x = K̃ + γ + ζ(x) with K̃ a Dirac point,
γ ∈ ℛ*_t and ‖ζ(x)‖₂ < W_t, so by the row-sum bound

    ‖Ĥ_q^{(τ,W,L)}‖₂ ≤ max_t sup_{‖ζ‖₂ ≤ W_t} ‖h̃_t(K_t + ζ) − E_D‖_∞ + 𝔍

for every q and L (the K′ valley gives the same supremum because
h̃_t(−k) = conj h̃_t(k)). The supremum is estimated on a polar grid of the disk
and multiplied by 1 + `margin`. This is not the constant of eq. (4.35), which
does not depend on W; it gives a finer KPM resolution ε_P at the same P.
"""
function spectral_bound(model::Model{T}, W; n::Integer = 24, margin::Real = 0.02, nᵣ::Integer = 16,
        n_φ::Integer = 48) where {T}
    stack = model.stack
    Wⱼ = triple(T, W)
    intra = zero(T)
    for t in 1:3, a in 1:nᵣ, b in 1:n_φ
        ζ = Wⱼ[t] * a / nᵣ * Vec2{T}(cospi(2b / n_φ), sinpi(2b / n_φ))
        h̃ = bloch(model.shells, stack, t, stack.K[t] + ζ; E_D = model.E_D)
        intra = max(intra, abs(h̃[1, 1]) + abs(h̃[1, 2]))
    end
    𝔍 = spectral_bound(model; n, margin)[2]
    return (1 + margin) * intra + 𝔍, 𝔍
end

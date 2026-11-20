# Labels of the reciprocal Hamiltonian and their truncation (manuscript,
# Sections 3, 4.3 and 4.4).
#
# A label x = (G, t α) of layer t has G = (G_s)_{s ≠ t} with G_s = B_s m_s,
# m_s ∈ ℤ². It is stored without the orbital index: `Label(t, m)` with
# m = (m₁, m₂, m₃) and m_t = 0, and it carries the two orbitals α = A, B.

"""
    Label(layer, m)

The pair of labels (G, t A), (G, t B) with t = `layer` and G_s = B_s m_s for the
two layers s ≠ t (`m[t]` is zero).
"""
struct Label
    layer::Int
    m::SVector{3,SVector{2,Int}}
end

"""The zero label (G = 0) of layer `t`: the pair (0, t A), (0, t B)."""
Label(t::Integer) = Label(t, zero(SVector{3,SVector{2,Int}}))

"""The label of layer `t` with integer coordinates `x`, `y` for the layers (k, l) = `others(t)`."""
function Label(t::Integer, x::SVector{2,Int}, y::SVector{2,Int})
    o = zero(SVector{2,Int})
    return Label(t, t == 1 ? SVector(o, x, y) : t == 2 ? SVector(x, o, y) : SVector(x, y, o))
end

# Order of the labels in a `Window`: by layer, then lexicographically by (m₁, m₂, m₃).
Base.isless(x::Label, y::Label) = isless((x.layer, x.m), (y.layer, y.m))

"""
    momentum(stack, q, x)

k_x = q + ∑_{s≠t} G_s, the momentum of the label `x` at the starting momentum `q`.
"""
@inline momentum(stack::Stack{T}, q::Vec2{T}, x::Label) where {T} =
    q + stack.B[1] * x.m[1] + stack.B[2] * x.m[2] + stack.B[3] * x.m[3]

"""
    shifted_momentum(stack, q, K̃, x)

ζ(x) = q − K̃_t + ∑_{s≠t} (B_s − B_t) m_s, eq. (4.7): the displacement of k_x from
the Dirac point K̃_t + B_t ∑_s m_s of layer t. `K̃` holds the nearest Dirac
points K̃_t of the three layers to q.
"""
@inline function shifted_momentum(stack::Stack{T}, q::Vec2{T}, K̃::SVector{3,Vec2{T}}, x::Label) where {T}
    t = x.layer
    k, l = others(t)
    return q - K̃[t] + (stack.B[k] - stack.B[t]) * x.m[k] + (stack.B[l] - stack.B[t]) * x.m[l]
end

"""‖G_s‖₂ for the two layers s ≠ t of a label; their maximum is compared with L in (4.27)."""
@inline function label_size(stack::Stack, x::Label)
    k, l = others(x.layer)
    return max(norm(stack.B[k] * x.m[k]), norm(stack.B[l] * x.m[l]))
end

"""
    Window(stack, q, W, L)

The labels of 𝒲*(q, W) ∩ ℒ*(L) at the starting momentum `q` (eqs. (4.8), (4.27)):
for each layer t, the labels x ∈ Ω*_t with

    ‖ζ(x)‖₂ < W_t      and      max(‖G_k‖₂, ‖G_l‖₂) < L_t.

`W` and `L` are numbers or triples (W₁, W₂, W₃), (L₁, L₂, L₃); `L = Inf` is not
allowed because 𝒲*(q, W) is infinite. The window of every layer is built by
direct enumeration, whether or not ‖q − K̃_t‖₂ < W_t, so it does not rely on the
connectedness of Theorem 4.3 and it contains the windows of layers whose zero
label is outside ([`search_window`](@ref) is the breadth-first alternative).

Fields: `q`; `W`, `L`; `K̃[t]` and `valley[t]`, the nearest Dirac point of layer
t to q and its valley; `labels` (sorted by layer, then integer coordinates);
`ζ[s]`, the shifted momentum of label s; `layers[t]`, the index range of layer
t; `index`, the position of each label.
"""
struct Window{T<:Real}
    q::Vec2{T}
    W::SVector{3,T}
    L::SVector{3,T}
    K̃::SVector{3,Vec2{T}}
    valley::SVector{3,Int}
    labels::Vector{Label}
    ζ::Vector{Vec2{T}}
    layers::NTuple{3,UnitRange{Int}}
    index::Dict{Label,Int}
end

"""`triple(T, x)`: a number `x` as (x, x, x), or the first three entries of `x`, with element type `T` (W and L per layer)."""
triple(::Type{T}, x::Real) where {T} = SVector{3,T}(x, x, x)
triple(::Type{T}, x) where {T} = SVector{3,T}(x[1], x[2], x[3])

function Window(stack::Stack{T}, q, W, L) where {T}
    q = Vec2{T}(q)
    W, L = triple(T, W), triple(T, L)
    all(>(0), W) && all(isfinite, L) && all(>(0), L) ||
        throw(ArgumentError("W must be positive and L positive and finite"))
    all(<(stack.Knorm / 2), W) || throw(ArgumentError("W must be smaller than ½‖K‖₂ = $(stack.Knorm / 2) Å⁻¹"))
    dirac = ntuple(t -> nearest_dirac(stack, t, q), 3)
    K̃ = SVector{3,Vec2{T}}(dirac[1][1], dirac[2][1], dirac[3][1])
    valley = SVector{3,Int}(dirac[1][2], dirac[2][2], dirac[3][2])
    labels = Label[]
    for t in 1:3
        enumerate_window!(labels, stack, t, q - K̃[t], W[t], L[t])
    end
    sort!(labels)
    return Window(stack, q, W, L, K̃, valley, labels)
end

"""
    Window(stack, q, W, L, K̃, valley, labels)

The window with the given sorted list of `labels` (no truncation is applied to it): fills in
`ζ`, `layers` and `index`. Used by the constructor above and for explicit label sets in the tests.
"""
function Window(stack::Stack{T}, q::Vec2{T}, W::SVector{3,T}, L::SVector{3,T}, K̃::SVector{3,Vec2{T}},
        valley::SVector{3,Int}, labels::Vector{Label}) where {T}
    ζ = [shifted_momentum(stack, q, K̃, x) for x in labels]
    layer = [x.layer for x in labels]
    layers = ntuple(t -> searchsortedfirst(layer, t):searchsortedlast(layer, t), 3)
    index = Dict{Label,Int}(x => s for (s, x) in enumerate(labels))
    return Window{T}(q, W, L, K̃, valley, labels, ζ, layers, index)
end

"""
    enumerate_window!(labels, stack, t, c, W, L) -> labels

Append the labels x of layer t with ‖c + C m_k + D m_l‖₂ < W and
max(‖B_k m_k‖₂, ‖B_l m_l‖₂) < L, where c = q − K̃_t, C = B_k − B_t, D = B_l − B_t,
so that c + C m_k + D m_l = ζ(x) (eq. (4.7)).

The enumeration is complete: ‖B_k m_k‖₂ < L gives ‖m_k‖_∞ ≤ L/σ_min(B_k), and for each m_k,
with e = c + C m_k, the condition ‖e + D m_l‖₂ < W gives ‖m_l + D⁻¹e‖₂ < W/σ_min(D), so m_l is
within ⌈W/σ_min(D)⌉ + 1 of the rounded point −D⁻¹e in each coordinate.
"""
function enumerate_window!(labels::Vector{Label}, stack::Stack{T}, t::Integer, c::Vec2{T}, W::T, L::T) where {T}
    k, l = others(t)
    B = stack.B
    C, D = B[k] - B[t], B[l] - B[t]
    σ_D = minimum(svdvals(Matrix(D)))
    σ_C = minimum(svdvals(Matrix(C)))
    σ_D > 0 || throw(ArgumentError("layers $t and $l have the same lattice; the stack is not incommensurate"))
    σ_C > 0 || throw(ArgumentError("layers $t and $k have the same lattice; the stack is not incommensurate"))
    xmax = ceil(Int, L / stack.σ)
    ymax = ceil(Int, W / σ_D) + 1
    D⁻¹ = inv(D)
    for x₂ in -xmax:xmax, x₁ in -xmax:xmax
        x = SVector(x₁, x₂)
        norm(B[k] * x) < L || continue
        e = c + C * x
        y₀ = round.(Int, -(D⁻¹ * e))
        for y₂ in -ymax:ymax, y₁ in -ymax:ymax
            y = y₀ + SVector(y₁, y₂)
            norm(e + D * y) < W || continue
            norm(B[l] * y) < L || continue
            push!(labels, Label(t, x, y))
        end
    end
    return labels
end

Base.length(window::Window) = length(window.labels)

"""Dimension 2 × (number of labels) of the truncated Hamiltonian."""
dimension(window::Window) = 2length(window)

"""The two orbital indices (A, B) of label number `s`."""
@inline orbitals(s::Integer) = (2s - 1, 2s)

"""
    zero_label(window, t)

Index of the zero label (0, t) of layer `t`, or 0 if it is not in the window,
that is, if ‖q − K̃_t‖₂ ≥ W_t (then the convention (4.10) sets the truncated
integrand to zero).
"""
zero_label(window::Window, t::Integer) = get(window.index, Label(t), 0)

"""The six steps S = {±e₁, ±e₂, ±(e₁ + e₂)}; B_t S are the shortest nonzero vectors of ℛ*_t."""
const STEPS = (SVector(1, 0), SVector(-1, 0), SVector(0, 1), SVector(0, -1), SVector(1, 1), SVector(-1, -1))

"""
    search_window(stack, q, W, t; L = Inf, limit = 10^7) -> Vector{Label}

The breadth-first search of Theorem 4.3(b) for layer `t`: starting from the
zero label, it moves to adjacent labels (one of G_k, G_l changed by a shortest
reciprocal vector of its lattice) and accepts a label x when

    ‖[k_x]_t − K̃‖₂ < W,      [y]_t = y − B_t ⌊B_t⁻¹ y⌋,

where K̃ ∈ {K_t, K′_t} is the Dirac point with ‖q − K̃ − γ‖₂ < W for some
γ ∈ ℛ*_t (the test is ℛ*_t-periodic in q). Under (4.12)–(4.14)
([`connectedness_range`](@ref)) the result is exactly 𝒲*_t(q, W). With a finite
`L`, labels with max(‖G_k‖₂, ‖G_l‖₂) ≥ L are not accepted; the result is then
the part of 𝒲*_t(q, W) ∩ ℒ*_t(L) connected to the zero label inside it, which
can be smaller than the set of [`Window`](@ref). Returns an empty list if the
zero label fails the test.
"""
function search_window(stack::Stack{T}, q, W::Real, t::Integer; L::Real = Inf, limit::Integer = 10^7) where {T}
    q = Vec2{T}(q)
    0 < W ≤ stack.Knorm / 2 || throw(ArgumentError("W must lie in (0, ½‖K‖₂]"))
    B = stack.B[t]
    K̃, valley = nearest_dirac(stack, t, q)
    K₀ = valley == 1 ? stack.K[t] : stack.K′[t]              # the representative in Γ*_t = B_t [0,1)²
    reduce(y) = y - B * floor.(B \ y)
    accept(x) = norm(reduce(momentum(stack, q, x)) - K₀) < W && label_size(stack, x) < L
    k, l = others(t)
    start = Label(t)
    found = Label[]
    accept(start) || return found
    seen = Set{Label}((start,))
    queue = [start]
    while !isempty(queue)
        x = popfirst!(queue)
        push!(found, x)
        length(found) ≤ limit || throw(ErrorException("the search exceeded $limit labels; give a finite L"))
        for s in STEPS, slot in (k, l)
            y = Label(t, setindex(x.m, x.m[slot] + s, slot))
            y in seen && continue
            push!(seen, y)
            accept(y) && push!(queue, y)
        end
    end
    return sort!(found)
end

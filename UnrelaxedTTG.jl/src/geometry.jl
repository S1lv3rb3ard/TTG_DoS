# Geometry of the three-layer stack (manuscript, Section 2): lattices,
# reciprocal lattices, sublattice offsets and Dirac points.
#
#   A_j        2×2 lattice basis of layer j (columns a₁, a₂), in Å
#   B_j        reciprocal basis, B_jᵀ A_j = 2π I, in Å⁻¹
#   τ_j        position of the B sublattice in layer j (the A sublattice is at 0)
#   K_j, K′_j  Dirac points B_j (2/3, 1/3) and B_j (1/3, 2/3)
#   (k, l)     the two layers other than j, in increasing order

"""Counterclockwise rotation by `θ` degrees: R(θ) = [cos θ  −sin θ; sin θ  cos θ]."""
rotation(θ::Real) = Mat2(cosd(θ), sind(θ), -sind(θ), cosd(θ))

"""`(k, l) = others(j)`: the two layers other than `j`, in increasing order."""
others(j::Integer) = j == 1 ? (2, 3) : j == 2 ? (1, 3) : (1, 2)

"""The layer other than `j` and `k`."""
third(j::Integer, k::Integer) = 6 - j - k

"""
    Stack(a, θ)

Three graphene layers with lattice constant `a` (Å) rotated counterclockwise by
the angles `θ = (θ₁, θ₂, θ₃)` (degrees). For each layer j

    A_j = R(θ_j) a [1 1/2; 0 √3/2],   B_j = 2π A_j⁻ᵀ,   τ_j = A_j (1/3, 1/3),
    K_j = B_j (2/3, 1/3),             K′_j = B_j (1/3, 2/3).

`|Γ*|` = |det B_j| is the area of a reciprocal unit cell and `Knorm` = ‖K‖₂ =
4π/(3a). Layers 1–2 and 2–3 are coupled; the coupling of the outer layers is
neglected (Section 2.2).
"""
struct Stack{T<:Real}
    a::T                                  # lattice constant (Å)
    θ::SVector{3,T}                       # layer angles (degrees, counterclockwise)
    R::SVector{3,Mat2{T}}                 # rotations R(θ_j)
    A::SVector{3,Mat2{T}}                 # lattice bases
    B::SVector{3,Mat2{T}}                 # reciprocal bases, BᵀA = 2πI
    τ::SVector{3,Vec2{T}}                 # B-sublattice offsets
    τ̂::SVector{3,Vec2{T}}                 # unit vectors τ_j/‖τ_j‖₂ (bond directions)
    K::SVector{3,Vec2{T}}                 # Dirac points K_j
    K′::SVector{3,Vec2{T}}                # Dirac points K′_j
    Γ::T                                  # |Γ*| = |det B_j| (Å⁻²)
    Knorm::T                              # ‖K‖₂ = 4π/(3a) (Å⁻¹)
    σ::T                                  # smallest singular value of B_j (Å⁻¹)
end

function Stack(a::Real, θ)
    a > 0 || throw(ArgumentError("the lattice constant a must be positive"))
    length(θ) == 3 || throw(ArgumentError("θ must hold the three layer angles (degrees)"))
    T = float(promote_type(typeof(a), eltype(θ)))
    A₀ = T(a) * Mat2{T}(1, 0, 1 / 2, √3 / 2)         # columns a₁ = (1, 0), a₂ = (1/2, √3/2)
    B₀ = 2π * inv(A₀)'
    τ₀ = A₀ * Vec2{T}(1 / 3, 1 / 3)
    K₀ = B₀ * Vec2{T}(2 / 3, 1 / 3)
    K′₀ = B₀ * Vec2{T}(1 / 3, 2 / 3)
    R = SVector{3}(rotation(T(θ[1])), rotation(T(θ[2])), rotation(T(θ[3])))
    τ = map(Rⱼ -> Rⱼ * τ₀, R)
    return Stack{T}(T(a), SVector{3,T}(θ[1], θ[2], θ[3]), R, map(Rⱼ -> Rⱼ * A₀, R), map(Rⱼ -> Rⱼ * B₀, R),
        τ, map(normalize, τ), map(Rⱼ -> Rⱼ * K₀, R), map(Rⱼ -> Rⱼ * K′₀, R), abs(det(B₀)), norm(K₀), minimum(svdvals(Matrix(B₀))))
end

"""Pairs of coupled layers (j, k) with j < k."""
const INTERFACES = ((1, 2), (2, 3))

"""The layers coupled to layer `j`."""
coupled(j::Integer) = j == 2 ? (1, 3) : (2,)

"""Orbital offset τ_{jα}: 0 for the A sublattice (α = 1), τ_j for B (α = 2)."""
orbital_offset(stack::Stack{T}, j::Integer, α::Integer) where {T} = α == 1 ? zero(Vec2{T}) : stack.τ[j]

"""Coordinates of the Dirac points in the reciprocal basis: K_t = B_t (2/3, 1/3), K′_t = B_t (1/3, 2/3)."""
const VALLEY = (SVector(2 // 3, 1 // 3), SVector(1 // 3, 2 // 3))

"""
    nearest_dirac(stack, t, q) -> (K̃, valley)

The Dirac point K̃_t of layer `t` nearest to `q` (Section 4.3), among

    𝒦_t = (K_t + ℛ*_t) ∪ (K′_t + ℛ*_t),

and its valley (+1 for K, −1 for K′). Ties are broken by a fixed rule: the K
valley before the K′ valley, and within a valley the first point in the
lexicographic order of the integer coordinates.
"""
function nearest_dirac(stack::Stack{T}, t::Integer, q) where {T}
    B = stack.B[t]
    c = B \ Vec2{T}(q)
    best, K̃, valley = T(Inf), zero(Vec2{T}), 0
    for (v, sign) in zip(VALLEY, (1, -1))
        n₀ = round.(Int, c - v)
        for d₁ in -1:1, d₂ in -1:1
            point = B * (n₀ + SVector(d₁, d₂) + v)
            distance = norm(Vec2{T}(q) - point)
            if distance < best
                best, K̃, valley = distance, point, sign
            end
        end
    end
    return K̃, valley
end

"""
    dirac_separation(stack, j, t)

β_jt = ‖K_j − K_t‖₂ = 2|sin((θ_t − θ_j)/2)| ‖K‖₂, eq. (4.11).
"""
dirac_separation(stack::Stack, j::Integer, t::Integer) = 2 * abs(sind((stack.θ[t] - stack.θ[j]) / 2)) * stack.Knorm

"""
    connectedness_range(stack, j) -> (lower, upper)

The interval of Theorem 4.3 for the window radius of layer `j`:

    β_{j,min} + (√3/2) β_{j,max} < W_j ≤ min(½‖K‖₂, (√3/2)(‖K‖₂ − β_{j,max}))

(conditions (4.12)–(4.14); the first upper bound is strict). For W_j in this
interval the window of layer j is connected by adjacent labels and is returned
by [`search_window`](@ref).
"""
function connectedness_range(stack::Stack, j::Integer)
    k, l = others(j)
    β = minmax(dirac_separation(stack, j, k), dirac_separation(stack, j, l))
    return β[1] + √3 / 2 * β[2], min(stack.Knorm / 2, √3 / 2 * (stack.Knorm - β[2]))
end

# Interlayer p_z hopping of graphene (manuscript, Appendix F; Fang & Kaxiras,
# PRB 93, 235153 (2016)) and its Fourier transform (F.8).

"""
    InterlayerParameters

The ten parameters of the interlayer function (Table F.2), with r = ‖x‖₂/a:

    V₀(r) = t₀ exp(−ξ₀ r²) cos(κ₀ r)
    V₃(r) = t₃ r² exp(−ξ₃ (r − x₃)²)
    V₆(r) = t₆ exp(−ξ₆ (r − x₆)²) sin(κ₆ r)

The amplitudes t₀, t₃, t₆ (eV) are called λ₀, λ₃, λ₆ in the manuscript; the
symbol λ is reserved for eigenvalues in this code.
"""
struct InterlayerParameters{T<:Real}
    t₀::T
    ξ₀::T
    κ₀::T
    t₃::T
    ξ₃::T
    x₃::T
    t₆::T
    ξ₆::T
    x₆::T
    κ₆::T
end

"""The parameters of Fang & Kaxiras (2016), Table F.2."""
const FANG_KAXIRAS = InterlayerParameters(0.3155, 1.7543, 2.0010, -0.0688, 3.4692, 0.5212,
    -0.0083, 2.8764, 1.5206, 1.5731)

"""Radial functions (V₀, V₃, V₆)(r) of (F.6), r = ‖x‖₂/a."""
@inline function radial(p::InterlayerParameters, r::Real)
    V₀ = p.t₀ * exp(-p.ξ₀ * r^2) * cos(p.κ₀ * r)
    V₃ = p.t₃ * r^2 * exp(-p.ξ₃ * (r - p.x₃)^2)
    V₆ = p.t₆ * exp(-p.ξ₆ * (r - p.x₆)^2) * sin(p.κ₆ * r)
    return V₀, V₃, V₆
end

@inline chebyshev₃(z) = 4z^3 - 3z
@inline chebyshev₆(z) = 32z^6 - 48z^4 + 18z^2 - 1

"""Sublattice sign σ_α: −1 for A (α = 1), +1 for B (α = 2)."""
@inline sublattice_sign(α::Integer) = α == 1 ? -1 : 1

"""
    angular(ê, τ̂ⱼ, τ̂ₖ, α, β) -> (Θ₃, Θ₆)

Angular factors of the pair (jα, kβ) in the unit direction `ê`,

    Θ₃ = σ_α T₃(ê·τ̂_j) − σ_β T₃(ê·τ̂_k),     Θ₆ = T₆(ê·τ̂_j) + T₆(ê·τ̂_k),

with T_n(cos φ) = cos nφ, σ_A = −1, σ_B = +1, and τ̂_j the unit vector along the
bond direction τ_j of layer j, for both sublattices. It is the convention of
the MATLAB code (`getInterlayerHopping.m`) and of `RelaxedTTG.jl`.

Relation to the manuscript. (F.7) is written in the direction r̂ of
r⃗ = r_{kβ} − r_{jα} = −x, where x = r_{jα} − r_{kβ} is the argument of h in (2.9):

    Θ_n(r̂) = T_n(−ς_α r̂·τ̂_{jα}) + T_n(ς_β r̂·τ̂_{kβ}).

Since T₃ is odd and T₆ is even, the factors above are (F.7) with ê = −r̂,
ς_α = σ_α and τ̂_{jα} = τ̂_j, that is, with ς_A = −1 and ς_B = +1. The manuscript
of 11 November 2026 prints ς_A = 1, ς_B = −1 and leaves τ̂_{jA} undefined
(τ_{jA} = 0); these are the manuscript points M1 of the code review of
11 November 2026, and the code follows the conventions of the two earlier codes.
"""
@inline function angular(ê::Vec2, τ̂ⱼ::Vec2, τ̂ₖ::Vec2, α::Integer, β::Integer)
    zⱼ = clamp(ê ⋅ τ̂ⱼ, -1, 1)
    zₖ = clamp(ê ⋅ τ̂ₖ, -1, 1)
    Θ₃ = sublattice_sign(α) * chebyshev₃(zⱼ) - sublattice_sign(β) * chebyshev₃(zₖ)
    Θ₆ = chebyshev₆(zⱼ) + chebyshev₆(zₖ)
    return Θ₃, Θ₆
end

"""
    interlayer_hopping(x, p, a, τ̂ⱼ, τ̂ₖ, α, β)

Interlayer hopping h_{kβ}^{jα}(x) (eV) at the in-plane separation
`x` = r_{jα} − r_{kβ} (Å), eq. (F.5):

    h = V₀(r) + V₃(r) Θ₃(x̂) + V₆(r) Θ₆(x̂),     r = ‖x‖₂/a.

It is real and satisfies h_{kβ}^{jα}(x) = h_{jα}^{kβ}(−x) (Assumption 2.6).
"""
@inline function interlayer_hopping(x::Vec2, p::InterlayerParameters, a::Real, τ̂ⱼ::Vec2, τ̂ₖ::Vec2,
        α::Integer, β::Integer)
    ρ = norm(x)
    V₀, V₃, V₆ = radial(p, ρ / a)
    ρ > 0 || return V₀                           # V₃(0) = V₆(0) = 0
    Θ₃, Θ₆ = angular(x / ρ, τ̂ⱼ, τ̂ₖ, α, β)
    return V₀ + V₃ * Θ₃ + V₆ * Θ₆
end

"""
    HankelTable(p, a, ρmax; R = 8a, nᵣ = 600, M = 384, tolerance = 1e-13)

Chebyshev interpolants on [0, ρmax] of the three Hankel transforms of (F.8),

    ℋ_n(ρ) = ∫₀^R V_n(s/a) J_n(ρ s) s ds,      n ∈ {0, 3, 6},

computed with an `nᵣ`-point Gauss–Legendre rule in s and interpolated at `M`
Chebyshev nodes in ρ. `R` is the real-space cutoff of the integral (the radial
functions decay like Gaussians in s/a; at s = 8a they are below 10⁻⁴⁰ eV).
`c[m, i]` is the coefficient of T_{m−1} for the order n = (0, 3, 6)[i]. Trailing
coefficients below `tolerance` times the largest coefficient are dropped (for
the default parameters the coefficients reach the rounding level 10⁻¹⁶ near
m = 60), which shortens the Clenshaw sum of [`hankel`](@ref).
"""
struct HankelTable{T<:Real}
    ρmax::T
    c::Matrix{T}
end

function HankelTable(p::InterlayerParameters, a::Real, ρmax::Real; R::Real = 8a, nᵣ::Integer = 600,
        M::Integer = 384, tolerance::Real = 1e-13)
    ρmax > 0 || throw(ArgumentError("ρmax must be positive"))
    T = float(promote_type(typeof(a), typeof(ρmax)))
    ξ, w = gauss_legendre(nᵣ)
    s = @. R * (ξ + 1) / 2
    ws = @. R * w / 2 * s
    V = [radial(p, sᵢ / a) for sᵢ in s]
    θ = [π * (i - 1 / 2) / M for i in 1:M]                 # Chebyshev nodes x_i = cos θ_i
    values = zeros(T, M, 3)
    for (i, θᵢ) in enumerate(θ)
        ρ = ρmax * (1 + cos(θᵢ)) / 2
        for (col, order) in enumerate((0, 3, 6))
            values[i, col] = sum(q -> ws[q] * V[q][col] * besselj(order, ρ * s[q]), eachindex(s))
        end
    end
    c = zeros(T, M, 3)
    for m in 1:M, col in 1:3                                 # discrete Chebyshev transform
        c[m, col] = (m == 1 ? 1 : 2) / M * sum(i -> values[i, col] * cos((m - 1) * θ[i]), 1:M)
    end
    floor = tolerance * maximum(abs, c)
    keep = something(findlast(m -> any(col -> abs(c[m, col]) > floor, 1:3), 1:M), 1)
    keep < M - 8 || @warn "the Chebyshev coefficients of the Hankel table have not decayed below $tolerance; increase M" maxlog = 1
    return HankelTable{T}(T(ρmax), c[1:keep, :])
end

"""
    hankel(table, ρ) -> (ℋ₀, ℋ₃, ℋ₆)

The three Hankel transforms at 0 ≤ `ρ` ≤ `table.ρmax` (Clenshaw summation).
"""
@inline function hankel(table::HankelTable{T}, ρ::Real) where {T}
    0 ≤ ρ ≤ table.ρmax * (1 + 8eps(T)) || throw(DomainError(ρ, "outside the tabulated range [0, $(table.ρmax)]"))
    x = clamp(2ρ / table.ρmax - 1, -one(T), one(T))
    c = table.c
    b₁ = b₂ = zero(SVector{3,T})
    @inbounds for m in size(c, 1):-1:2
        b₁, b₂ = SVector(c[m, 1], c[m, 2], c[m, 3]) + 2x * b₁ - b₂, b₁
    end
    @inbounds h = SVector(c[1, 1], c[1, 2], c[1, 3]) + x * b₁ - b₂
    return h[1], h[2], h[3]
end

"""
    fourier_hopping(table, ξ, τ̂ⱼ, τ̂ₖ) -> 2×2 SMatrix

Fourier transform (2.10) of the interlayer hopping, eq. (F.8): entry (α, β) is

    ĥ_{kβ}^{jα}(ξ) = (2π)⁻¹ [ℋ₀(‖ξ‖₂) + i Θ₃(ξ̂) ℋ₃(‖ξ‖₂) − Θ₆(ξ̂) ℋ₆(‖ξ‖₂)]      (eV Å²),

with the angular factors of the pair (jα, kβ) in the direction of ξ. The
factors i = (−i)³ and −1 = (−i)⁶ come from the transform of e^{±inφ} (Appendix F).
"""
@inline function fourier_hopping(table::HankelTable{T}, ξ::Vec2, τ̂ⱼ::Vec2, τ̂ₖ::Vec2) where {T}
    ρ = norm(ξ)
    ℋ₀, ℋ₃, ℋ₆ = hankel(table, ρ)
    ξ̂ = ρ > 0 ? ξ / ρ : zero(ξ)                    # ℋ₃(0) = ℋ₆(0) = 0
    entry(α, β) = begin
        Θ₃, Θ₆ = angular(ξ̂, τ̂ⱼ, τ̂ₖ, α, β)
        complex(ℋ₀ - Θ₆ * ℋ₆, Θ₃ * ℋ₃) / (2 * T(π))
    end
    return SMatrix{2,2,Complex{T},4}(entry(1, 1), entry(2, 1), entry(1, 2), entry(2, 2))
end

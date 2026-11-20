# Interlayer p_z hopping of graphene (manuscript, Appendix F; Fang & Kaxiras,
# PRB 93, 235153 (2016)) and its Fourier transform (F.8). Two radial models:
# the functions of Fang & Kaxiras (`InterlayerParameters`, `FANG_KAXIRAS`) and an
# analytic refit of their threefold and sixfold terms whose Fourier transform
# decays like a Gaussian (`AnalyticRefit`, `ANALYTIC_REFIT`).

"""
    InterlayerModel

Supertype of the radial models of the interlayer hopping (F.5),

    h(x) = V₀(r) + V₃(r) Θ₃(x̂) + V₆(r) Θ₆(x̂),      r = ‖x‖₂/a.

A model `p` defines `radial(p, r) -> (V₀, V₃, V₆)`; the angular factors Θ₃, Θ₆ are
those of [`angular`](@ref) for every model. Available: [`InterlayerParameters`](@ref)
(the form of Fang & Kaxiras) and [`AnalyticRefit`](@ref); by name, see
[`INTERLAYER_MODELS`](@ref).
"""
abstract type InterlayerModel end

"""
    InterlayerParameters

The ten parameters of the interlayer function (Table F.2), with r = ‖x‖₂/a:

    V₀(r) = t₀ exp(−ξ₀ r²) cos(κ₀ r)
    V₃(r) = t₃ r² exp(−ξ₃ (r − x₃)²)
    V₆(r) = t₆ exp(−ξ₆ (r − x₆)²) sin(κ₆ r)

The amplitudes t₀, t₃, t₆ (eV) are called λ₀, λ₃, λ₆ in the manuscript; the
symbol λ is reserved for eigenvalues in this code.
"""
struct InterlayerParameters{T<:Real} <: InterlayerModel
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

"""
    AnalyticRefit(t₀, ξ₀, κ₀, β₃, v₃, β₆, v₆)

Interlayer hopping with the isotropic function V₀ of Fang & Kaxiras and analytic
threefold and sixfold functions (note "A refit of the Fang–Kaxiras interlayer
hopping with a Gaussian-decaying Fourier transform", 17 November 2026):

    V₀(r) = t₀ exp(−ξ₀ r²) cos(κ₀ r)
    Ṽ_m(r) = r^m exp(−β_m r²/2) Σ_j v_{m,j} L_j^{(m)}(β_m r²),      m ∈ {3, 6},

with the generalised Laguerre polynomials L_j^{(m)} and `v₃`, `v₆` the tuples of
coefficients (eV). Ṽ_m(r) cos(m(φ − φ₀)) is real-analytic on ℝ², so the hopping is
real-analytic and its Fourier transform decays like a Gaussian. The Hankel
transforms of Ṽ₃, Ṽ₆ are known in closed form ([`closed_form_hankel`](@ref)).
"""
struct AnalyticRefit{T<:Real,N₃,N₆} <: InterlayerModel
    t₀::T
    ξ₀::T
    κ₀::T
    β₃::T
    v₃::NTuple{N₃,T}
    β₆::T
    v₆::NTuple{N₆,T}
end

"""
The analytic refit of the note of 17 November 2026: V₀ of Fang & Kaxiras, and the
L²(r dr)-best approximations of their V₃ (three terms) and V₆ (four terms) in the
form of [`AnalyticRefit`](@ref). It differs from `FANG_KAXIRAS` by at most
0.33 meV in V₃ and 0.061 meV in V₆; it is a fit to the published Fang–Kaxiras
functions, not to DFT data.
"""
const ANALYTIC_REFIT = AnalyticRefit(0.3155, 1.7543, 2.0010,
    6.2595, (-0.505794, 0.0978994, -0.0100687),
    3.9995, (-0.0372218, 0.00134815, 0.00141368, -0.000468766))

"""
The analytic refit of orders (J₃, J₆) = (6, 7): seven Laguerre terms for V₃ and eight for V₆,
L²(r dr)-optimal widths rounded to four decimals, coefficients (the projection at the rounded
width) to six significant digits. It differs from `FANG_KAXIRAS` by at most 0.158 meV in V₃ and
0.0288 meV in V₆; the Hankel transforms decay like exp(−ρ²a²/(2β₃)), β₃ = 8.6136. On the line cut of
θ₄ the momentum LDoS differs from Fang–Kaxiras by at most 0.24 % (report of 18 November 2026,
Section 3). A fit to the published Fang–Kaxiras functions, not to DFT data.
"""
const ANALYTIC_REFIT_6_7 = AnalyticRefit(0.3155, 1.7543, 2.0010,
    8.6136, (-0.803373, 0.284918, -0.0875052, 0.0228126, -0.00598521, 0.00118628, -0.000385441),
    5.3768, (-0.0916305, 0.0202163, -0.00197556, -0.00129062, 0.000795897, -0.000305824, 7.05733e-05, -1.78057e-05))

"""
The interlayer models by name, as the drivers in `examples/` accept them:
`:fang_kaxiras` (`FANG_KAXIRAS`, the published functions of Fang & Kaxiras),
`:analytic_refit` (`ANALYTIC_REFIT`, orders (2, 3)) and `:analytic_refit_6_7`
(`ANALYTIC_REFIT_6_7`, orders (6, 7), the model of the manuscript and the default, see
`DEFAULT_INTERLAYER`).
"""
const INTERLAYER_MODELS = (fang_kaxiras = FANG_KAXIRAS, analytic_refit = ANALYTIC_REFIT,
                           analytic_refit_6_7 = ANALYTIC_REFIT_6_7)

"""
Name of the default interlayer model, `:analytic_refit_6_7` (`ANALYTIC_REFIT_6_7`), the model
adopted for the manuscript (20 November 2026). Until version 0.5 the default was
`:fang_kaxiras`; pass `interlayer = :fang_kaxiras` (or `parameters = FANG_KAXIRAS`) for it.
"""
const DEFAULT_INTERLAYER = :analytic_refit_6_7

"""
    interlayer_model(name::Symbol) -> InterlayerModel

The model `INTERLAYER_MODELS[name]`; an `ArgumentError` for an unknown name.
"""
function interlayer_model(name::Symbol)
    haskey(INTERLAYER_MODELS, name) ||
        throw(ArgumentError("unknown interlayer model :$name; use one of $(join(":" .* string.(keys(INTERLAYER_MODELS)), ", "))"))
    return INTERLAYER_MODELS[name]
end

"""
    laguerre_sum(v, m, x)

Σ_j v[j+1] L_j^{(m)}(x) for the coefficients `v` (a tuple), by the three-term recurrence
j L_j = (2j − 1 + m − x) L_{j−1} − (j − 1 + m) L_{j−2}, L₀ = 1, L₁ = 1 + m − x.
"""
@inline function laguerre_sum(v::NTuple{N,T}, m::Integer, x::Real) where {N,T}
    L₀ = one(x)
    s = v[1] * L₀
    N == 1 && return s
    L₁ = 1 + m - x
    s += v[2] * L₁
    for j in 2:N-1
        L₀, L₁ = L₁, ((2j - 1 + m - x) * L₁ - (j - 1 + m) * L₀) / j
        s += v[j+1] * L₁
    end
    return s
end

"""Radial functions (V₀, Ṽ₃, Ṽ₆)(r) of the analytic refit, r = ‖x‖₂/a."""
@inline function radial(p::AnalyticRefit, r::Real)
    V₀ = p.t₀ * exp(-p.ξ₀ * r^2) * cos(p.κ₀ * r)
    r² = r^2
    V₃ = r * r² * exp(-p.β₃ * r² / 2) * laguerre_sum(p.v₃, 3, p.β₃ * r²)
    V₆ = r² * r² * r² * exp(-p.β₆ * r² / 2) * laguerre_sum(p.v₆, 6, p.β₆ * r²)
    return V₀, V₃, V₆
end

"""
    closed_form_hankel(p, a, ρ) -> (ℋ₃, ℋ₆) or nothing

The Hankel transforms ℋ_n(ρ) = ∫₀^∞ V_n(s/a) J_n(ρs) s ds of the threefold and sixfold
functions in closed form, where one is known: for [`AnalyticRefit`](@ref),

    ℋ_m(ρ) = a² β_m^{−m−1} (aρ)^m exp(−(aρ)²/(2β_m)) Σ_j (−1)^j v_{m,j} L_j^{(m)}((aρ)²/β_m)

(Lemma 1 of the note of 17 November 2026). `nothing` for the Fang–Kaxiras form, whose
transforms are computed by quadrature.
"""
closed_form_hankel(::InterlayerParameters, a::Real, ρ::Real) = nothing

function closed_form_hankel(p::AnalyticRefit, a::Real, ρ::Real)
    ϱ² = (a * ρ)^2
    transform(β, v, m) = a^2 * β^(-m - 1) * sqrt(ϱ²)^m * exp(-ϱ² / (2β)) *
                         laguerre_sum(ntuple(j -> (-1)^(j - 1) * v[j], length(v)), m, ϱ² / β)
    return transform(p.β₃, p.v₃, 3), transform(p.β₆, p.v₆, 6)
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
@inline function interlayer_hopping(x::Vec2, p::InterlayerModel, a::Real, τ̂ⱼ::Vec2, τ̂ₖ::Vec2,
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

computed with an `nᵣ`-point Gauss–Legendre rule in s (ℋ₃ and ℋ₆ in closed form
where [`closed_form_hankel`](@ref) gives them) and interpolated at `M`
Chebyshev nodes in ρ. `R` is the real-space cutoff of the integral (the radial
functions decay like Gaussians in s/a; at s = 8a they are below 10⁻⁴⁰ eV).
`c[m, i]` is the coefficient of T_{m−1} for the order n = (0, 3, 6)[i]. Trailing
coefficients below `tolerance` times the largest coefficient are dropped (for
the τ = 8.5 of the examples 53 rows are kept with the default refit and 54 with
Fang–Kaxiras), which shortens the Clenshaw sum of [`hankel`](@ref).
"""
struct HankelTable{T<:Real}
    ρmax::T
    c::Matrix{T}
end

function HankelTable(p::InterlayerModel, a::Real, ρmax::Real; R::Real = 8a, nᵣ::Integer = 600,
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
        exact = closed_form_hankel(p, a, ρ)
        for (col, order) in enumerate((0, 3, 6))
            exact === nothing || col == 1 || continue        # ℋ₃, ℋ₆ in closed form: no quadrature
            values[i, col] = sum(q -> ws[q] * V[q][col] * besselj(order, ρ * s[q]), eachindex(s))
        end
        exact === nothing || (values[i, 2:3] .= exact)
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

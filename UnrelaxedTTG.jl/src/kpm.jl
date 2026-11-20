# Densities from Chebyshev moments (manuscript, Section 4.5).
#
# For a Hermitian matrix A with σ(sA) ⊂ (−1, 1), the P Chebyshev moments
# μ_n = ∑_c [T_n(sA)]_{cc}, n = 0, …, P − 1, of a set of start orbitals c give a regularised
# density ∑_c [k(E, A)]_{cc}. Two regularisation kernels k use the same moments:
#
# `Jackson()` (default), the kernel polynomial method of the manuscript,
#
#   ∑_c [δ^P_KPM(E, A)]_{cc} = s/(π √(1 − (sE)²)) ∑_{n=0}^{P−1} g_n^P μ_n T_n(sE)      (4.33)
#
# with the Jackson coefficients g_n^P of (4.34); the scale is s = (1 − δ_s)/B_H with the bound B_H
# of (4.35), and ε_P = π/(sP) is the nominal resolution;
#
# `DeltaChebyshev(η, m)`, the high-order regularized delta-Chebyshev method of Yi, Massatt,
# Horning, Luskin, Pixley and Kaye (arXiv:2512.03149), with a rational kernel of order m at the
# smoothing parameter η;
#
# `Gaussian(ε)`, the truncated Chebyshev expansion of the Gaussian δ_ε of Section 2.5 of the
# manuscript (KPM error note of 16 November 2026, Section 4).

"""
    Kernel

The regularisation of a density computed from Chebyshev moments, chosen with the keyword `kernel`
of [`kpm_density`](@ref), [`momentum_ldos`](@ref), [`moire_dos`](@ref) and [`brillouin_dos`](@ref):

- [`Jackson`](@ref)`()`, the default: the Jackson kernel of the manuscript, eqs. (4.33), (4.34);
- [`DeltaChebyshev`](@ref)`(η, m)`: the rational kernel of order m of arXiv:2512.03149;
- [`Gaussian`](@ref)`(ε)`: the Gaussian δ_ε of width ε, expanded in Chebyshev polynomials.

The moments μ_n are the same for all of them, so a stored set of moments can be evaluated with any.
"""
abstract type Kernel end

"""
    Jackson()

The Jackson kernel of the kernel polynomial method (manuscript, eqs. (4.33), (4.34)): the density
is s/(π√(1 − (sE)²)) ∑_n g_n^P μ_n T_n(sE) with the coefficients [`jackson`](@ref). It is positive,
and its resolution ε_P = π/(sP) is fixed by the number P of moments.
"""
struct Jackson <: Kernel end

"""
    DeltaChebyshev(η, m)

The rational kernel of order m ≥ 1 at the smoothing parameter η > 0 (eV) of the high-order
regularized delta-Chebyshev method (Yi, Massatt, Horning, Luskin, Pixley, Kaye, arXiv:2512.03149,
eq. (15)),

    K_η(E, x) = −(1/π) ∑_{l=1}^m Im[ w_l / (E − x + η z_l) ] = η⁻¹ K((E − x)/η),

with the equispaced poles z_l and the weights w_l of Colbrook, Horning and Townsend (SIAM Rev. 63
(2021) 489, eqs. (5.8), (5.7)):

    z_l = 2l/(m + 1) − 1 + i,      ∑_{l=1}^m w_l z_l^j = δ_{j0},   j = 0, …, m − 1.

The solution of this system is w_l = ∏_{k≠l} z_k/(z_k − z_l), the value at 0 of the Lagrange basis
polynomial of the node z_l (apply ∑_l p(z_l) ℓ_l(0) = p(0) to p(z) = z^j); the weights are computed
from this product. Fields: `η`, `z`, `w`. `K(E, x)` evaluates K_η(E, x) for a kernel `K`.

K = K₁(·, 0) is a kernel of order m (loc. cit., Definition 5.1): ∫K = 1, ∫K(u) u^j du = 0 for
0 < j < m, and |K(u)| ≤ C/(1 + |u|)^{m+1}. For a measure with a density that is m times continuously
differentiable near E, the regularised density converges to the density at E as O(η^m log(1/η))
(loc. cit., Theorem 5.2), where the Jackson kernel gives O(P⁻²). For m = 1, K_η is the Lorentzian
η/(π((E − x)² + η²)). For m ≥ 3 the kernel takes negative values, so the regularised density need
not be positive.

The density is evaluated from P moments by [`kpm_density`](@ref); the number of moments that a
given η needs is measured by [`truncation_bound`](@ref).
"""
struct DeltaChebyshev <: Kernel
    η::Float64
    z::Vector{ComplexF64}
    w::Vector{ComplexF64}
end

function DeltaChebyshev(η::Real, m::Integer)
    η > 0 || throw(ArgumentError("the smoothing parameter η must be positive"))
    m ≥ 1 || throw(ArgumentError("the order m must be a positive integer"))
    z = [complex(2l / (m + 1) - 1, 1.0) for l in 1:m]
    w = [prod(z[k] / (z[k] - z[l]) for k in 1:m if k ≠ l; init = complex(1.0)) for l in 1:m]
    return DeltaChebyshev(η, z, w)
end

(K::DeltaChebyshev)(E::Real, x::Real) = -sum(imag(w / (E - x + K.η * z)) for (z, w) in zip(K.z, K.w)) / π

Base.show(io::IO, K::DeltaChebyshev) = print(io, "DeltaChebyshev(", K.η, ", ", length(K.z), ")")

"""
    Gaussian(ε)

The Gaussian δ_ε(E − x) = exp(−(E − x)²/(2ε²))/(ε√(2π)) of width ε > 0 (eV) of Section 2.5 of the
manuscript, the regularisation in which the momentum LDoS and its error estimate are stated. The
density is the truncated Chebyshev expansion of δ_ε(E − A) itself (KPM error note of 16 November
2026, revised 18 November, Section 4):

    δ_ε(E − x/s) = ∑_{n≥0} c_n(E) T_n(x),   c_n(E) = ((2 − δ_{n0})/π) ∫_0^π δ_ε(E − cos θ/s) cos nθ dθ,

for x ∈ [−1, 1], so that ∑_c [δ_ε(E − A)]_{cc} ≈ ∑_{n<P} c_n(E) μ_n ([`kpm_density`](@ref)). The
left-out terms are bounded by [`truncation_bound`](@ref) (Proposition 5 of the note); they decay
like exp(−(Psε)²/2), so P sε ≥ 4 is enough in practice (Table 2 of the note: the bound is
1.6 × 10⁻⁴ of the peak value 1/(ε√(2π)) for P sε = 4). Unlike the Jackson kernel, whose width is fixed
by P and which differs from δ_ε by about 2.7 % of the peak however large P is, ε is chosen
independently of P, and the error tends to zero as P grows at fixed ε.

The truncated expansion is not a positive kernel: the density can be negative where the exact
regularised density is small, by at most the truncation bound. Expanding a Gaussian-smoothed delta
function in orthogonal polynomials is not new: Lin, Saad and Yang (SIAM Rev. 58 (2016) 34,
Section 3.1.3) expand it in Legendre polynomials (Delta–Gauss–Legendre). Field: `ε`. `K(E, x)`
evaluates δ_ε(E − x).
"""
struct Gaussian <: Kernel
    ε::Float64
    function Gaussian(ε::Real)
        ε > 0 || throw(ArgumentError("the width ε must be positive"))
        return new(ε)
    end
end

(K::Gaussian)(E::Real, x::Real) = exp(-(E - x)^2 / (2 * K.ε^2)) / (K.ε * sqrt(2π))

Base.show(io::IO, K::Gaussian) = print(io, "Gaussian(", K.ε, ")")

"""
    jackson(P) -> g

Jackson coefficients of eq. (4.34), g[n+1] = g_n^P for n = 0, …, P − 1:

    g_n^P = (2 − δ_{n0}) [(P − n + 1) cos(πn/(P+1)) + sin(πn/(P+1)) cot(π/(P+1))]/(P + 1).
"""
function jackson(P::Integer)
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    return [(n == 0 ? 1 : 2) * ((P - n + 1) * cospi(n / (P + 1)) + sinpi(n / (P + 1)) * cot(π / (P + 1))) / (P + 1)
            for n in 0:P-1]
end

"""
    kpm_resolution(s, P[, kernel]) -> ε

The parameter (eV) that sets the energy resolution of the density from P moments at the scale s:
the nominal resolution ε_P = π/(sP) of the Jackson kernel (Section 4.5; default), the smoothing
parameter η of a [`DeltaChebyshev`](@ref) kernel, and the width ε of a [`Gaussian`](@ref) kernel;
the last two do not depend on P. They are not widths of the same kind. A Gaussian of width ε has
the full width at half maximum 2.35ε. The full width at half maximum of the kernel is 2.38 ε_P for Jackson
(computed at the centre of the scaled interval for P = 4000) and 2η, 1.46η, 1.13η, 0.92η, 0.77η,
0.66η for the delta-Chebyshev kernels of order m = 1, …, 6; the kernels of Jackson with P moments
and of order 6 with η = 3.58 ε_P = 11.2/(sP) have the same width.
"""
kpm_resolution(s::Real, P::Integer) = π / (s * P)
kpm_resolution(s::Real, P::Integer, ::Jackson) = kpm_resolution(s, P)
kpm_resolution(::Real, ::Integer, K::DeltaChebyshev) = K.η
kpm_resolution(::Real, ::Integer, K::Gaussian) = K.ε

"""
    kpm_density!(ρ, μ, s, E; kernel = Jackson()) -> ρ
    kpm_density(μ, s, E; kernel = Jackson()) -> ρ

The regularised density ∑_c [k(E, A)]_{cc} (1/eV) at the energies `E` (eV) from the moments
`μ[n+1]` = μ_n = ∑_c [T_n(sA)]_{cc}, n = 0, …, P − 1, P = `length(μ)`.

`kernel = Jackson()`: the KPM density (4.33),

    ρ(E) = s/(π √(1 − x²)) ∑_{n=0}^{P−1} g_n^P μ_n T_n(x),      x = sE,

with T_n(x) generated by T_{n+1} = 2x T_n − T_{n−1}. Every |sE| must be below 1.

`kernel = DeltaChebyshev(η, m)`: the density of arXiv:2512.03149, eq. (18), truncated after P terms,

    ρ(E) = ∑_{n=0}^{P−1} ν_n(E) μ_n,      K_η(E, x/s) = ∑_{n≥0} ν_n(E) T_n(x),   x ∈ [−1, 1],

which approximates ∑_c [K_η(E, A)]_{cc}. The coefficients ν_n are used in closed form. For ζ outside
[−1, 1], with r = i√(1 − ζ²) and ϱ = ζ − r = 1/(ζ + r),

    1/(ζ − x) = (1/r) [1 + 2 ∑_{n≥1} ϱⁿ T_n(x)],      x ∈ [−1, 1],

and |ϱ| < 1 for Im ζ > 0 with the principal square root. With ζ_l = s(E + η z_l) this gives

    ρ(E) = −(s/π) Im ∑_{l=1}^m (w_l/r_l) [μ_0 + 2 ∑_{n=1}^{P−1} ϱ_lⁿ μ_n].

(The reference computes the same coefficients with a fast cosine transform.) The terms decay like
|ϱ_l|ⁿ ≈ exp(−n sη/√(1 − (sE)²)), so P must be large compared with 1/(sη):
[`truncation_bound`](@ref) bounds the part of the series that is left out, and the difference to
the density from fewer moments, `view(μ, 1:P′)`, measures it (the reference chooses P this way).
`E` is not restricted.

`kernel = Gaussian(ε)`: the truncated Chebyshev expansion of the Gaussian δ_ε,

    ρ(E) = ∑_{n=0}^{P−1} c̃_n(E) μ_n,      δ_ε(E − x/s) = ∑_{n≥0} c_n(E) T_n(x),   x ∈ [−1, 1],

which approximates ∑_c [δ_ε(E − A)]_{cc}. The coefficients
c̃_n(E) = ((2 − δ_{n0})/M) ∑_j δ_ε(E − x_j/s) cos nθ_j are those of the discrete cosine transform at
the M = 2P Chebyshev nodes x_j = cos θ_j, θ_j = π(j + 1/2)/M, j = 0, …, M − 1; they differ from
c_n(E) only by aliasing of the coefficients c_m with m > 3P. The sum is evaluated as

    ρ(E) = (1/M) ∑_j δ_ε(E − x_j/s) g_j,      g_j = μ_0 + 2 ∑_{n=1}^{P−1} μ_n T_n(x_j),

with g_j computed once for all energies and only at the nodes with |E − x_j/s| ≤ 40ε for some E
(at the other nodes δ_ε underflows to zero). [`truncation_bound`](@ref) bounds the terms left out
and the aliasing together; P sε ≥ 4 makes both small. `E` is not restricted.
"""
kpm_density!(ρ::AbstractVector, μ::AbstractVector, s::Real, E; kernel::Kernel = Jackson()) =
    kpm_density!(ρ, μ, s, E, kernel)

kpm_density(μ::AbstractVector, s::Real, E; kernel::Kernel = Jackson()) =
    kpm_density!(zeros(length(E)), μ, s, E, kernel)

function kpm_density!(ρ::AbstractVector, μ::AbstractVector, s::Real, E, ::Jackson)
    g = jackson(length(μ))
    for (i, Eᵢ) in enumerate(E)
        x = s * Eᵢ
        abs(x) < 1 || throw(DomainError(Eᵢ, "the energy lies outside the scaled interval |E| < $(1 / s)"))
        T₀, T₁ = one(x), x
        total = g[1] * μ[1]
        for n in 2:length(μ)
            total += g[n] * μ[n] * T₁
            T₀, T₁ = T₁, 2x * T₁ - T₀
        end
        ρ[i] = total * s / (π * sqrt(1 - x^2))
    end
    return ρ
end

"""
    resolvent_series(ζ) -> (r, ϱ)

r = i√(1 − ζ²) and ϱ = 1/(ζ + r) = ζ − r of the expansion 1/(ζ − x) = (1/r)[1 + 2 ∑_{n≥1} ϱⁿ T_n(x)],
x ∈ [−1, 1], for Im ζ > 0; then |ϱ| < 1 (ζ = cos(a + ib) with 0 < a < π and b < 0 gives
√(1 − ζ²) = sin(a + ib), whose real part is positive, and ϱ = e^{−i(a + ib)}, |ϱ| = e^b).
"""
function resolvent_series(ζ::Complex)
    r = im * sqrt(1 - ζ^2)
    return r, inv(ζ + r)
end

function kpm_density!(ρ::AbstractVector, μ::AbstractVector, s::Real, E, K::DeltaChebyshev)
    P = length(μ)
    for (i, Eᵢ) in enumerate(E)
        total = zero(ComplexF64)
        for (z, w) in zip(K.z, K.w)
            r, ϱ = resolvent_series(s * (Eᵢ + K.η * z))
            series = zero(ϱ)                              # ∑_{n=1}^{P−1} ϱⁿ μ_n, by Horner's rule
            for n in P:-1:2
                series = ϱ * (series + μ[n])
            end
            total += w / r * (μ[1] + 2series)
        end
        ρ[i] = -s * imag(total) / π
    end
    return ρ
end

"""
    gaussian_nodes(P) -> M

Number M = 2P of Chebyshev nodes x_j = cos θ_j, θ_j = π(j + 1/2)/M, j = 0, …, M − 1, at which
[`kpm_density`](@ref) samples δ_ε for the kernel [`Gaussian`](@ref).
"""
gaussian_nodes(P::Integer) = 2P

# Nodes j with |E − x_j/s| ≤ 40ε, for one energy: outside them exp(−(E − x_j/s)²/(2ε²)) < e⁻⁸⁰⁰
# underflows to 0.0, so leaving them out of the sum changes nothing. One node is added on each side
# against rounding in acos.
function gaussian_window(s::Real, ε::Real, E::Real, M::Integer)
    x₋, x₊ = s * (E - 40ε), s * (E + 40ε)
    (x₋ > 1 || x₊ < -1) && return 1:0
    θ₋, θ₊ = acos(min(x₊, 1.0)), acos(max(x₋, -1.0))         # θ is decreasing in x
    j₋ = max(0, floor(Int, M * θ₋ / π - 1 / 2) - 1)
    j₊ = min(M - 1, ceil(Int, M * θ₊ / π - 1 / 2) + 1)
    return j₋:j₊
end

function kpm_density!(ρ::AbstractVector, μ::AbstractVector, s::Real, E, K::Gaussian)
    P = length(μ)
    M = gaussian_nodes(P)
    windows = [gaussian_window(s, K.ε, Eᵢ, M) for Eᵢ in E]
    used = filter(!isempty, windows)
    # g_j = μ_0 + 2 ∑_{n=1}^{P−1} μ_n T_n(x_j) at the nodes that some energy needs
    j₋ = isempty(used) ? 0 : minimum(first, used)
    j₊ = isempty(used) ? -1 : maximum(last, used)
    g = zeros(j₊ - j₋ + 1)
    for j in j₋:j₊
        x = cospi((j + 1 / 2) / M)
        T₀, T₁ = one(x), x
        total = 0.0
        for n in 2:P
            total += μ[n] * T₁
            T₀, T₁ = T₁, 2x * T₁ - T₀
        end
        g[j-j₋+1] = μ[1] + 2total
    end
    for (i, Eᵢ) in enumerate(E)
        total = 0.0
        for j in windows[i]
            total += K(Eᵢ, cospi((j + 1 / 2) / M) / s) * g[j-j₋+1]
        end
        ρ[i] = total / M
    end
    return ρ
end

"""
    truncation_bound(K::DeltaChebyshev, s, E, P) -> b

A bound for the terms n ≥ P that [`kpm_density`](@ref) with the kernel `K` leaves out, per unit of
μ_0: for moments μ_n = ∑_c a_c [T_n(sA)]_{cc} with weights a_c ≥ 0 and σ(sA) ⊂ [−1, 1], so that
|μ_n| ≤ μ_0,

    |∑_c a_c [K_η(E, A)]_{cc} − ρ_P(E)| ≤ μ_0 b(E),
    b(E) = (2s/π) ∑_{l=1}^m |w_l| |ϱ_l|^P / (|r_l| (1 − |ϱ_l|)),

with r_l, ϱ_l of ζ_l = s(E + η z_l) as in [`kpm_density`](@ref) (the geometric series of the
left-out terms). `b` has the unit 1/eV; for a collection `E` it has one entry per energy. For the momentum LDoS
(4.36), μ_0/6 ≤ 1; for the densities of states, μ_0 is the field `mass` of the result. The bound
is for exact moments; it does not include the rounding errors of the recurrence. It uses only
|μ_n| ≤ μ_0 and was 150 to 600 times the measured truncation error for the moments of a line cut
(16 November 2026, set 2, W = 0.3771, L = 45.84, m = 6, η = 12 and 16 meV, P = 4000 against 8000).
"""
function truncation_bound(K::DeltaChebyshev, s::Real, E::Real, P::Integer)
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    b = 0.0
    for (z, w) in zip(K.z, K.w)
        r, ϱ = resolvent_series(s * (E + K.η * z))
        b += abs(w) * abs(ϱ)^P / (abs(r) * (1 - abs(ϱ)))
    end
    return 2s * b / π
end

truncation_bound(K::DeltaChebyshev, s::Real, E, P::Integer) = [truncation_bound(K, s, Eᵢ, P) for Eᵢ in E]

"""
    gaussian_tail_bound(s, ε, E, P) -> R

The bound R_P(E) ≥ ∑_{n≥P} |c_n(E)| of Proposition 5 of the KPM error note (16 November 2026,
revised 18 November), for |sE| < 1 (otherwise, and when the formula overflows, `Inf`):

    R_P(E) = (2s/π) [2/(cosh(Ps²ε²) √((1 − |sE|)(3 + |sE|))) + (√(π/2)/(sε)) exp(−(1 − |sE|)²/(8s²ε²))]
             × 1/(1 − exp(−Ps²ε²)) × exp(sinh²(Ps²ε²)/(2s²ε²) − P²s²ε²).
"""
function gaussian_tail_bound(s::Real, ε::Real, E::Real, P::Integer)
    abs(s * E) < 1 || return Inf
    sE = abs(s * E)
    Ps²ε² = P * s^2 * ε^2
    bracket = 2 / (cosh(Ps²ε²) * sqrt((1 - sE) * (3 + sE))) +
              sqrt(π / 2) / (s * ε) * exp(-(1 - sE)^2 / (8 * s^2 * ε^2))
    R = 2s / π * bracket / (-expm1(-Ps²ε²)) * exp(sinh(Ps²ε²)^2 / (2 * s^2 * ε^2) - Float64(P)^2 * s^2 * ε^2)
    return isnan(R) ? Inf : R                  # 0 × Inf when cosh and sinh overflow: no bound
end

"""
    truncation_bound(K::Gaussian, s, E, P) -> b

A bound for the difference between ∑_c a_c [δ_ε(E − A)]_{cc} and the density from P moments of
[`kpm_density`](@ref) with the kernel `K = Gaussian(ε)`, per unit of μ_0, for weights a_c ≥ 0 and
σ(sA) ⊂ [−1, 1] (so |μ_n| ≤ μ_0). Two parts:

- the terms n ≥ P that are left out, at most ∑_{n≥P} |c_n(E)|;
- the sampling of δ_ε at M = 2P nodes: the computed coefficients are c̃_n = c_n + ∑_{k≥1} (−1)^k
  (c_{2kM−n} + c_{2kM+n}), so ∑_{n<P} |c̃_n − c_n| ≤ ∑_{m>3P} |c_m| (Remark 7(i) of the note).

Each tail ∑_{n≥Q} |c_n(E)| is bounded by R_{Q′}(E) of Proposition 5 ([`gaussian_tail_bound`](@ref))
for any Q′ ≤ Q, because the tail decreases with Q; R grows with Q′ once Q′s²ε² exceeds about 1.1,
so the smaller of R_Q and R_{Q′} with Q′ = min(Q, ⌊1.09/(s²ε²)⌋) is used. `b` has the unit 1/eV and
is `Inf` for |sE| ≥ 1, where Proposition 5 does not apply; for a collection `E` it has one entry
per energy. For the momentum LDoS (4.36), μ_0/6 ≤ 1; for the densities of states, μ_0 is the field
`mass` of the result. The bound is for exact moments; it does not include the rounding errors of
the recurrence (Remark 7(iii) of the note: about 0.51 Δμ/ε for errors Δμ in the moments).
"""
function truncation_bound(K::Gaussian, s::Real, E::Real, P::Integer)
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    ε = K.ε
    Q′(Q) = Int(min(Q, max(1.0, floor(1.09 / (s^2 * ε^2)))))
    tail(Q) = min(gaussian_tail_bound(s, ε, E, Q), gaussian_tail_bound(s, ε, E, Q′(Q)))
    left_out = tail(P)
    return left_out + min(left_out, tail(3P + 1))
end

truncation_bound(K::Gaussian, s::Real, E, P::Integer) = [truncation_bound(K, s, Eᵢ, P) for Eᵢ in E]

"""
    chebyshev_moments(H̃, orbitals, P) -> μ

Moments μ[n+1] = ∑_c [T_n(H̃)]_{cc}, n = 0, …, P − 1, for the start orbitals c of
`orbitals`, where H̃ = sA is Hermitian with spectrum in (−1, 1). With
t_m = T_m(H̃) e_c, the doubling relations

    μ_{2m} = 2⟨t_m, t_m⟩ − μ₀,      μ_{2m+1} = 2⟨t_{m+1}, t_m⟩ − μ₁

give two moments per matrix–vector product. All start vectors are advanced
together as the columns of a dense matrix.
"""
function chebyshev_moments(H̃::AbstractMatrix{<:Number}, orbitals::AbstractVector{<:Integer}, P::Integer)
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    n, c = size(H̃, 1), length(orbitals)
    μ = zeros(P)
    c == 0 && return μ
    previous = zeros(eltype(H̃), n, c)                     # t₀
    for (column, orbital) in enumerate(orbitals)
        previous[orbital, column] = 1
    end
    current = H̃ * previous                                # t₁
    μ[1] = c
    P ≥ 2 && (μ[2] = real(dot(previous, current)))
    m = 1
    while 2m + 1 ≤ P                                      # here previous = t_{m−1}, current = t_m
        μ[2m+1] = 2 * real(dot(current, current)) - μ[1]
        2m + 2 ≤ P || break
        mul!(previous, H̃, current, 2, -1)                # t_{m+1} = 2H̃ t_m − t_{m−1}, in place of t_{m−1}
        previous, current = current, previous
        μ[2m+2] = 2 * real(dot(current, previous)) - μ[2]
        m += 1
    end
    return μ
end

"""
    reference_moments(H̃, orbitals, P) -> μ

The same moments from the plain recurrence T_{n+1} = 2H̃ T_n − T_{n−1}, one
moment per product (reference for [`chebyshev_moments`](@ref)).
"""
function reference_moments(H̃::AbstractMatrix, orbitals::AbstractVector{<:Integer}, P::Integer)
    Y = zeros(eltype(H̃), size(H̃, 1), length(orbitals))
    for (c, orbital) in enumerate(orbitals)
        Y[orbital, c] = 1
    end
    T₀, T₁ = Y, H̃ * Y
    μ = zeros(P)
    μ[1] = real(dot(Y, T₀))
    P ≥ 2 && (μ[2] = real(dot(Y, T₁)))
    for n in 3:P
        T₀, T₁ = T₁, 2 * (H̃ * T₁) - T₀
        μ[n] = real(dot(Y, T₁))
    end
    return μ
end

"""
    row_sum_norm(H)

‖H‖_∞ = max_i ∑_j |H_ij| of a Hermitian sparse matrix, computed as the largest column sum of
|H_ij| (the two agree because H = H*). It bounds the spectral norm: ‖H‖₂ ≤ ‖H‖_∞.
"""
function row_sum_norm(H::SparseMatrixCSC)
    values = nonzeros(H)
    return maximum(column -> sum(p -> abs(values[p]), nzrange(H, column); init = 0.0), axes(H, 2); init = 0.0)
end

"""
    scaled(H, s, B_H) -> s·H

The scaled matrix H̃ = sH of the recurrence (a new matrix; `H` is not changed), after the check
‖H‖_∞ ≤ B_H ([`row_sum_norm`](@ref)). Since ‖H‖₂ ≤ ‖H‖_∞, the check gives
σ(sH) ⊂ [−(1 − δ_s), 1 − δ_s] ⊂ (−1, 1) for s = (1 − δ_s)/B_H, which the Chebyshev recurrence
needs. It is made for every matrix because the constants of [`spectral_bound`](@ref) are grid
estimates of suprema, not bounds. Throws an `ErrorException` if the check fails.
"""
function scaled(H::SparseMatrixCSC, s::Real, B_H::Real)
    bound = row_sum_norm(H)
    bound ≤ B_H || throw(ErrorException("‖Ĥ_q‖_∞ = $bound eV exceeds B_H = $B_H eV; pass a larger B_H"))
    return s * H
end

"""
    gpu_matrix(S::SlicedELL) -> S on the GPU
    gpu_matrix(H̃) -> H̃ on the GPU
    gpu_matrix(H̃, T) -> H̃ on the GPU with element type T

Copy of the scaled Hermitian sparse matrix H̃ = sĤ_q^{(τ,W,L)} on the GPU.
[`chebyshev_moments`](@ref) applied to the result runs the recurrence of Section 4.5 on the GPU
with the same doubling relations as on the CPU.

The first method uploads the sliced ELLPACK storage [`SlicedELL`](@ref) of H̃ (element type and
stored rows as chosen there); the recurrence then uses the kernel of the extension. This is the
fast path, used by the drivers:

    S = SlicedELL(H̃, ComplexF32; reach = c)              # on the CPU; c = start orbitals
    μ = chebyshev_moments(gpu_matrix(S), c, P)

The other two methods upload H̃ in compressed-row storage (`CuSparseMatrixCSR`); the recurrence
then uses the CUSPARSE product. It is slower (measured 2.5 to 3 times at the settings of the line
cuts) and its single-precision results vary from run to run in the last digits; it is kept as an
independent implementation for the tests.

The one-argument method for a `SparseMatrixCSC` keeps the element type of H̃ (`ComplexF64` for the matrices of
[`hamiltonian`](@ref)). The two-argument method converts the entries to `T` and the indices to
`Int32` first; `T = ComplexF32` runs the recurrence in single precision. The moments are returned
as `Float64` in both cases, but with `ComplexF32` every product and inner product is accumulated in
single precision (measured on 16 November 2026: moments differ from the CPU by 10⁻⁶ and the LDoS
by 1–2 × 10⁻⁷ in the measure (5.2); in double precision by 10⁻¹⁴).

Defined when CUDA.jl is loaded (package extension `UnrelaxedTTGCUDAExt`); without CUDA.jl the
function has no methods.
"""
function gpu_matrix end

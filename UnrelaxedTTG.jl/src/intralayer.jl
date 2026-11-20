# Intralayer hopping of graphene (manuscript, Appendix F): Jung–MacDonald
# neighbour shells, the Bloch matrix (2.4) and the Dirac energy (Section 2.4).

"""
    jung_macdonald(table = :I) -> (tA, tB)

Wannier π-band hoppings (eV) of Jung & MacDonald, PRB 87, 195450 (2013).
`tA[n+1]` is the amplitude t_n^A of the A–A shell n = 0, …, 7 (`tA[1]` is the
on-site energy) and `tB[n]` the amplitude t_n^B of the A–B shell n = 1, …, 10.
`:I` is their Table I, which is Table F.1 of the manuscript; `:II` is their
Table II, which the MATLAB production code `TTG_DoS` and `RelaxedTTG.jl` use.
"""
function jung_macdonald(table::Symbol = :I)
    table === :I && return (
        [0.3208, 0.22378, 0.04813, -0.02402, 0.00263, 0.00111, 0.00018, -0.00008],
        [-2.92181, -0.27897, 0.02669, -0.00885, -0.01772, 0.00675, -0.00262, 0.00019,
            -0.00068, -0.00237])
    table === :II && return (
        [0.3302, 0.23206, 0.04969, -0.02499, 0.00285, 0.00204, -0.00014, -0.00029],
        [-2.99251, -0.28983, 0.02791, -0.00877, -0.01870, 0.00621, -0.00256, -0.00018,
            -0.00033, -0.00264])
    throw(ArgumentError("unknown table $table; use :I or :II"))
end

"""
    Shells(a; table = :I, tA, tB)

The neighbour shells of Appendix F for the lattice constant `a`, in the frame
of an unrotated layer:

    rA[i], tAᵢ[i]   the sites r ∈ S_n^A (same sublattice; r = 0 is the on-site term)
                    and their amplitudes t_n^A,
    rB[i], tBᵢ[i]   the sites r ∈ S_n^B ⊂ ℛ + τ_B (other sublattice) and t_n^B.

Shells are found by enumerating lattice sites and sorting by distance, so the
n-th amplitude sits on the n-th distinct distance d(n, β). No cutoff is applied:
the model has finite range. With the bond length a/√3 the shell radii are
d(n, A) = a √(0, 1, 3, 4, 7, 9, 12, 13)[n+1] and d(n, B) = (a/√3) √(1, 4, 7, 13, 16, 19, 25, 28, 31, 37)[n],
so for a = 1.42√3 Å the largest bond is d(7, A) = √13 a ≈ 8.87 Å (d(10, B) ≈ 8.64 Å).

The other fields: `dA`, `dB` are the shell radii and `tA`, `tB` the amplitudes
per shell, as given to the constructor.
"""
struct Shells{T<:Real}
    a::T                          # lattice constant (Å)
    rA::Vector{Vec2{T}}
    tAᵢ::Vector{T}
    rB::Vector{Vec2{T}}
    tBᵢ::Vector{T}
    dA::Vector{T}                 # shell radii d(n, A), n = 0, 1, …
    dB::Vector{T}                 # shell radii d(n, B), n = 1, 2, …
    tA::Vector{T}                 # amplitudes per shell
    tB::Vector{T}
end

function Shells(a::T; table::Symbol = :I, tA = jung_macdonald(table)[1], tB = jung_macdonald(table)[2]) where {T<:AbstractFloat}
    A₀ = a * Mat2{T}(1, 0, 1 / 2, √3 / 2)
    τ₀ = A₀ * Vec2{T}(1 / 3, 1 / 3)
    reach = 12
    cells = [SVector(m₁, m₂) for m₂ in -reach:reach for m₁ in -reach:reach]
    dA, rA, tAᵢ = neighbour_shells([A₀ * m for m in cells], T.(tA), a, reach)
    dB, rB, tBᵢ = neighbour_shells([A₀ * m + τ₀ for m in cells], T.(tB), a, reach)
    return Shells{T}(a, rA, tAᵢ, rB, tBᵢ, dA, dB, T.(tA), T.(tB))
end

"""
    neighbour_shells(sites, t, a, reach) -> (radii, r, amplitude)

The sites of the first `length(t)` distinct distances among `sites`, with the amplitude of their
shell: `radii[n]` is the n-th distinct distance, and `r[i]` is a site at one of these distances
with amplitude `amplitude[i]` = `t[n]`. Two distances are the same shell when they differ by less
than max(10⁻⁹, 1000 eps(T)) a: 10⁻⁹ a in Float64, 1.2·10⁻⁴ a in Float32, where rounding moves
equal distances apart by ~10⁻⁶ a (distinct shells of the patch differ by more than 10⁻² a). `sites` must be the points of a lattice patch of `reach` cells in each direction;
an `ArgumentError` is thrown unless the requested shells lie inside its inscribed disk, of radius
(√3/2) `reach` a, so that every shell is complete.
"""
function neighbour_shells(sites::Vector{Vec2{T}}, t::Vector{T}, a::T, reach::Integer) where {T}
    tolerance = max(1e-9, 1000 * eps(T)) * a
    d = norm.(sites)
    sorted = sort(d)
    radii = [sorted[i] for i in eachindex(sorted) if i == 1 || sorted[i] - sorted[i-1] > tolerance]
    # Shells are complete only inside the inscribed disk of the enumerated patch.
    complete = reach * a * √3 / 2
    length(radii) ≥ length(t) && radii[length(t)] < complete - tolerance ||
        throw(ArgumentError("the enumerated patch does not contain $(length(t)) complete shells"))
    resize!(radii, length(t))
    r, amplitude = Vec2{T}[], T[]
    for (n, radius) in enumerate(radii), i in eachindex(sites)
        abs(d[i] - radius) < tolerance && (push!(r, sites[i]); push!(amplitude, t[n]))
    end
    return radii, r, amplitude
end

"""
    bloch(shells, stack, t, k; E_D = 0) -> 2×2 Hermitian SMatrix

Bloch matrix h̃_t(k) of layer `t` (eqs. (2.4), (F.3)), with the on-site term
shifted by −`E_D` (Section 2.4):

    h̃^{A}_{A}(k) = h̃^{B}_{B}(k) = ∑_n t_n^A ∑_{r ∈ S_n^A} e^{−ik·R(θ_t) r} − E_D,
    h̃^{B}_{A}(k) = conj h̃^{A}_{B}(k) = ∑_n t_n^B ∑_{r ∈ S_n^B} e^{−ik·R(θ_t) r}.

Rows and columns are ordered (A, B); entry (α, β) is h̃^{α}_{β}(k).
"""
function bloch(shells::Shells{T}, stack::Stack{T}, t::Integer, k; E_D::Real = zero(T)) where {T}
    κ = stack.R[t]' * Vec2{T}(k)                  # k·R(θ_t) r = (R(θ_t)ᵀ k)·r
    d = zero(T)
    @inbounds for i in eachindex(shells.rA)       # S_n^A = −S_n^A, so the sum is real
        d += shells.tAᵢ[i] * cos(κ ⋅ shells.rA[i])
    end
    f = zero(Complex{T})
    @inbounds for i in eachindex(shells.rB)
        f += shells.tBᵢ[i] * cis(-(κ ⋅ shells.rB[i]))
    end
    d -= E_D
    return SMatrix{2,2,Complex{T},4}(d, f, conj(f), d)      # column-major: (A,A), (B,A), (A,B), (B,B)
end

"""
    dirac_energy(shells)

Dirac energy E_D (Section 2.4): the common diagonal value of the Bloch matrix
at a Dirac point, E_D = ∑_n t_n^A ∑_{r ∈ S_n^A} cos(K·r). It does not depend on
the lattice constant or on the layer.
"""
function dirac_energy(shells::Shells{T}) where {T}
    B₀ = 2π * inv(shells.a * Mat2{T}(1, 0, 1 / 2, √3 / 2))'
    K = B₀ * Vec2{T}(2 / 3, 1 / 3)
    return sum(i -> shells.tAᵢ[i] * cos(K ⋅ shells.rA[i]), eachindex(shells.rA))
end

"""
    intralayer_bound(shells; E_D = 0)

Row-sum bound ∑_n |t_n^A||S_n^A| + ∑_n |t_n^B||S_n^B| + |E_D| ≥ sup_k ‖h̃_t(k)‖₂
of the first term of B_H, eq. (4.35) (the on-site term is counted separately
from E_D, so the bound holds with and without the shift).
"""
intralayer_bound(shells::Shells; E_D::Real = 0) = sum(abs, shells.tAᵢ) + sum(abs, shells.tBᵢ) + abs(E_D)

"""
    band_gap(shells, stack, r; E_D = dirac_energy(shells), n = 96)

Estimate of the band-gap function φ(r) of eq. (4.24),

    φ(r) = inf{|λ| : λ ∈ σ(h̃_t(k)), dist(k, 𝒦_t) ≥ r},

for the shifted model, by sampling k on an n × n grid of one reciprocal cell of
layer 2 (φ does not depend on the layer). A sampled infimum is an upper
estimate of φ(r), not a bound.
"""
function band_gap(shells::Shells{T}, stack::Stack{T}, r::Real; E_D::Real = dirac_energy(shells), n::Integer = 96) where {T}
    φ = T(Inf)
    for i₂ in 0:n-1, i₁ in 0:n-1
        k = stack.B[2] * Vec2{T}(i₁ / n, i₂ / n)
        K̃, _ = nearest_dirac(stack, 2, k)
        norm(k - K̃) ≥ r || continue
        h = bloch(shells, stack, 2, k; E_D)
        d, f = real(h[1, 1]), abs(h[2, 1])
        φ = min(φ, abs(d + f), abs(d - f))
    end
    return φ
end

# Momentum LDoS and density of states (manuscript, Sections 4.1, 4.5, 4.6).

"""
    kpm_scale(model, E; δ_s = 0.05, B_H = nothing, kernel = Jackson()) -> (s, B_H)

The KPM scale s = (1 − δ_s)/B_H of Section 4.5, with B_H from
[`spectral_bound`](@ref) unless given. For the Jackson kernel, whose density (4.33) is defined
only there, it checks |E| < B_H/(1 − δ_s) for the energies `E`; the delta-Chebyshev and Gaussian
densities are defined for every E.
"""
function kpm_scale(model::Model, E; δ_s::Real = 0.05, B_H::Union{Nothing,Real} = nothing, kernel::Kernel = Jackson())
    0 < δ_s < 1 || throw(ArgumentError("δ_s must lie in (0, 1)"))
    bound = B_H === nothing ? spectral_bound(model)[1] : float(B_H)
    s = (1 - δ_s) / bound
    kernel isa Jackson && !(maximum(abs, E) * s < 1) &&
        throw(DomainError(maximum(abs, E), "the energies must satisfy |E| < $(1 / s) eV"))
    return s, bound
end

"""Moments of the start orbitals in batches of at most `batch` columns."""
function batched_moments(H̃, start::Vector{Int}, P::Integer, batch::Integer)
    μ = zeros(P)
    for part in Iterators.partition(start, batch)
        μ .+= chebyshev_moments(H̃, part, P)
    end
    return μ
end

"""
A piecewise-linear momentum path: points `q` (Å⁻¹), arc length `s` of every
point, and the arc lengths `ticks` and `labels` of its vertices.
"""
struct Linecut{T<:Real}
    q::Vector{Vec2{T}}
    s::Vector{T}
    ticks::Vector{T}
    labels::Vector{String}
end

"""
    linecut(vertices, Nq; labels) -> Linecut

The path through `vertices` with `Nq` steps per segment.
"""
function linecut(vertices::AbstractVector{<:Vec2{T}}, Nq::Integer; labels = string.(eachindex(vertices))) where {T}
    q, s, ticks = [vertices[1]], [zero(T)], [zero(T)]
    for i in 1:length(vertices)-1
        a, b = vertices[i], vertices[i+1]
        for step in 1:Nq
            push!(q, a + (b - a) * step / Nq)
            push!(s, ticks[end] + norm(b - a) * step / Nq)
        end
        push!(ticks, ticks[end] + norm(b - a))
    end
    return Linecut{T}(q, s, ticks, collect(String, labels))
end

"""
    linecut(stack, Nq; Γ = :moire) -> Linecut

The path K₁–K₂–Γ₁₂–Γ₂₃–K₂–K₃ of Figures 5.1 and 5.2. Γ_jk is the centre of the
moiré Brillouin zone of the pair (j, k), the hexagon with adjacent corners K_j
and K_k: it lies on the perpendicular bisector of K_jK_k at distance
(√3/2)‖K_k − K_j‖₂ from the midpoint, so ‖Γ_jk − K_j‖₂ = ‖Γ_jk − K_k‖₂ = ‖K_k − K_j‖₂.
`Γ = :legacy` puts it at distance ‖K_k − K_j‖₂ from the midpoint, which is
not the zone centre (`getLinecut(K, Nq, 'legacy')` in TTG_DoS, described there
as the construction of Figures 4 and 5 of arXiv:2606.13434).
"""
function linecut(stack::Stack, Nq::Integer; Γ::Symbol = :moire)
    Γ in (:moire, :legacy) || throw(ArgumentError("Γ must be :moire or :legacy"))
    K = stack.K
    centre(j, k) = begin
        edge = K[k] - K[j]
        (K[j] + K[k]) / 2 + rotation(-90) * edge * (Γ === :moire ? √3 / 2 : 1)
    end
    return linecut([K[1], K[2], centre(1, 2), centre(2, 3), K[2], K[3]], Nq;
        labels = ["K₁", "K₂", "Γ₁₂", "Γ₂₃", "K₂", "K₃"])
end

"""
    momentum_ldos(model, q, E; W, L, P, δ_s = 0.05, B_H = nothing, kernel = Jackson()) -> result

The computed momentum LDoS of eq. (4.36) at the momenta `q` (a vector of
momenta or a [`Linecut`](@ref)) and energies `E` (eV):

    D̂_P^{(τ,W,L)}(q, E) = (1/3) ∑_j (1/2) ∑_α [δ^P_KPM(E, Ĥ_q^{(τ,W,L)})]_{(0,jα),(0,jα)}.

A zero label that is not in the window contributes zero (convention (4.10)).
The window is rebuilt for every q. The momenta are distributed over the
available threads.

With `kernel = DeltaChebyshev(η, m)` the regularisation δ^P_KPM is replaced by the rational kernel
K_η of order m, evaluated from the same P moments ([`kpm_density`](@ref),
[`truncation_bound`](@ref)); with `kernel = Gaussian(ε)` by the Gaussian δ_ε itself, expanded in
Chebyshev polynomials, so that the result approximates the Gaussian LDoS of Section 2.5 up to
`truncation_bound`; `kernel = Jackson()` is (4.36).

`result`: `E`, `q`, `ldos` (momenta × energies), `μ` (momenta × P, summed over
the zero-label orbitals present), `s`, `B_H`, `kernel`, `resolution` (ε_P for the Jackson kernel,
η for `DeltaChebyshev`, ε for `Gaussian`; [`kpm_resolution`](@ref)), `labels` (number of labels per q) and `present`
(number of zero labels in the window).
"""
function momentum_ldos(model::Model{T}, q::AbstractVector, E; W, L, P::Integer, δ_s::Real = 0.05,
        B_H::Union{Nothing,Real} = nothing, kernel::Kernel = Jackson()) where {T}
    s, bound = kpm_scale(model, E; δ_s, B_H, kernel)
    μ = zeros(length(q), P)
    ldos = zeros(length(q), length(E))
    labels, present = zeros(Int, length(q)), zeros(Int, length(q))
    Threads.@threads for i in eachindex(q)
        window = Window(model.stack, q[i], W, L)
        start = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
        H̃ = scaled(hamiltonian(model, window), s, bound)
        μ[i, :] = chebyshev_moments(H̃, start, P)
        kpm_density!(view(ldos, i, :), view(μ, i, :), s, E, kernel)
        labels[i], present[i] = length(window), length(start) ÷ 2
    end
    ldos ./= 6
    return (E = collect(E), q = collect(q), ldos = ldos, μ = μ, s = s, B_H = bound, kernel = kernel,
        resolution = kpm_resolution(s, P, kernel), labels = labels, present = present)
end

momentum_ldos(model::Model, path::Linecut, E; kwargs...) = momentum_ldos(model, path.q, E; kwargs...)

"""
    moire_cell(stack, j; layer2 = 1) -> (M, k)

The matrix M_j = B_k − B_j of the moiré reciprocal lattice of the pair (j, k)
(Section 4.6), with the coupled layer k = k(j): k(1) = k(3) = 2 and
k(2) = `layer2` ∈ {1, 3}.
"""
function moire_cell(stack::Stack, j::Integer; layer2::Integer = 1)
    layer2 in (1, 3) || throw(ArgumentError("layer2 must be 1 or 3"))
    k = j == 2 ? layer2 : 2
    return stack.B[k] - stack.B[j], k
end

"""
    moire_dos(model, E; W, L, N, P, δ_s = 0.05, B_H = nothing, kernel = Jackson(), layer2 = 1, batch = 64,
              verbose = false) -> result

The moiré-cell rule of eq. (4.39) for the density of states,

    ν* ∑_j |C_j|/N² ∑_{K̃ ∈ {K_j, K′_j}} ∑_α ∑_{n ∈ {0,…,N−1}²} ∑_{m: ‖r_n + M_j m‖₂ < W_j}
        [δ^P_KPM(E, Ĥ_{K̃ + r_n}^{(τ,W,L)})]_{x_m x_m},

with r_n = M_j n/N, the centre labels x_m = ((G_k = B_k m, G_l = 0), jα), and
ν* = (∑_j |𝒜_j||Γ*_j|)⁻¹ = 1/(6|Γ*|). The mask ‖r_n + M_j m‖₂ < W_j is the test
‖ζ(x_m)‖₂ < W_j and depends on n. By Proposition 4.10 the rule is a Riemann sum
of the integral of the truncated integrand over the disks ‖q − K̃‖₂ < W_j; its
hypothesis √3 β_jk ≤ W_j is checked.

A centre label with ‖B_k m‖₂ ≥ L_j is not in ℒ*(L) and is skipped; the number of
skipped centres is returned in `missing` (increase L until it is 0).

`kernel = DeltaChebyshev(η, m)` evaluates the same weighted moments with the rational kernel K_η of
order m in place of δ^P_KPM ([`kpm_density`](@ref)), and `kernel = Gaussian(ε)` with the Gaussian
δ_ε expanded in Chebyshev polynomials; the weights of the rule are positive, so
[`truncation_bound`](@ref) applies with μ₀ = `mass`.

`result`: `E`, `ρ`, `μ` (weighted moments), `s`, `B_H`, `kernel`, `resolution` (ε_P, η or ε,
[`kpm_resolution`](@ref)), `mass` =
μ₀ (the weight of the sampled centres), `disk_mass` = ν* · 4π ∑_j W_j² (its limit
as N → ∞), `point_moments[g]` (N² × P, one array per layer and valley, in the
order j = 1, 2, 3 and K, K′), `centres`, `missing`, `labels` (largest window).
"""
function moire_dos(model::Model{T}, E; W, L, N::Integer, P::Integer, δ_s::Real = 0.05,
        B_H::Union{Nothing,Real} = nothing, kernel::Kernel = Jackson(), layer2::Integer = 1, batch::Integer = 64,
        verbose::Bool = false) where {T}
    N ≥ 1 || throw(ArgumentError("N must be a positive integer"))
    stack = model.stack
    Wⱼ, Lⱼ = triple(T, W), triple(T, L)
    s, bound = kpm_scale(model, E; δ_s, B_H, kernel)
    ν = 1 / (6 * stack.Γ)
    grid = [SVector(n₁, n₂) for n₂ in 0:N-1 for n₁ in 0:N-1]
    groups = [(j, valley) for j in 1:3 for valley in (1, -1)]
    point_moments = [zeros(N^2, P) for _ in groups]
    centres, missing_centres, largest = zeros(Int, length(groups)), zeros(Int, length(groups)), zeros(Int, length(groups))
    μ = zeros(P)
    for (g, (j, valley)) in enumerate(groups)
        M, k = moire_cell(stack, j; layer2)
        √3 * dirac_separation(stack, j, k) ≤ Wⱼ[j] || throw(ArgumentError(
            "Proposition 4.10 needs √3 β_jk ≤ W_j for layer $j: $(√3 * dirac_separation(stack, j, k)) > $(Wⱼ[j])"))
        K̃ = valley == 1 ? stack.K[j] : stack.K′[j]
        M⁻¹ = inv(M)
        reach = ceil(Int, Wⱼ[j] / minimum(svdvals(Matrix(M)))) + 1
        centre_count, skipped, window_size = zeros(Int, N^2), zeros(Int, N^2), zeros(Int, N^2)
        Threads.@threads for i in eachindex(grid)
            r = M * grid[i] / N
            window = Window(stack, K̃ + r, Wⱼ, Lⱼ)
            norm(window.K̃[j] - K̃) < 1e-9 * stack.Knorm ||
                throw(ErrorException("the nearest Dirac point of layer $j to K̃ + r_n is not K̃"))
            m₀ = round.(Int, -(M⁻¹ * r))
            start = Int[]
            for d₂ in -reach:reach, d₁ in -reach:reach
                m = m₀ + SVector(d₁, d₂)
                norm(r + M * m) < Wⱼ[j] || continue                       # the mask ‖ζ(x_m)‖₂ < W_j
                index = get(window.index, Label(j, setindex(zero(SVector{3,SVector{2,Int}}), m, k)), 0)
                index == 0 ? (skipped[i] += 1) : append!(start, orbitals(index))
            end
            H̃ = scaled(hamiltonian(model, window), s, bound)
            point_moments[g][i, :] = batched_moments(H̃, start, P, batch)
            centre_count[i], window_size[i] = length(start) ÷ 2, length(window)
        end
        w = ν * abs(det(M)) / N^2
        μ .+= w .* vec(sum(point_moments[g]; dims = 1))
        centres[g], missing_centres[g], largest[g] = sum(centre_count), sum(skipped), maximum(window_size)
        verbose && @printf("moiré-cell DoS: layer %d, valley %+d done (%d centres, %d skipped, at most %d labels)\n",
            j, valley, centres[g], missing_centres[g], largest[g])
    end
    sum(missing_centres) == 0 || @warn "$(sum(missing_centres)) centre labels lie outside ℒ*(L) and were skipped; " *
        "the corresponding part of the disks is not sampled. Increase L."
    return (E = collect(E), ρ = kpm_density(μ, s, E; kernel), μ = μ, s = s, B_H = bound, kernel = kernel,
        resolution = kpm_resolution(s, P, kernel), mass = μ[1], disk_mass = ν * 4π * sum(abs2, Wⱼ), point_moments = point_moments, centres = centres,
        missing = missing_centres, labels = maximum(largest))
end

"""
    brillouin_dos(model, E; W, L, N, P, δ_s = 0.05, B_H = nothing, kernel = Jackson()) -> result

The Brillouin-zone quadrature 𝒯_N of eq. (4.1) applied to the truncated
operator, D̂_P^{(N,τ,W,L)}(E) with KPM in place of δ_ε:

    (ν*/N²) ∑_j |Γ*_j| ∑_α ∑_{q ∈ 𝔖_j(N)} [δ^P_KPM(E, Ĥ_q^{(τ,W,L)})]_{(0,jα),(0,jα)},

𝔖_j(N) = {B_j n/N : n ∈ {0,…,N−1}²}. By the convention (4.10) only the grid
points with ‖q − K̃_j‖₂ < W_j contribute. This grid is fine on the atomic scale
and coarse on the moiré scale; [`moire_dos`](@ref) is the rule used for the
figures. `kernel` as in [`moire_dos`](@ref). `result`: `E`, `ρ`, `μ`, `s`, `B_H`, `kernel`,
`resolution`, `mass` = μ₀, `points` (number of contributing grid points per layer).
"""
function brillouin_dos(model::Model{T}, E; W, L, N::Integer, P::Integer, δ_s::Real = 0.05,
        B_H::Union{Nothing,Real} = nothing, kernel::Kernel = Jackson()) where {T}
    stack = model.stack
    Wⱼ, Lⱼ = triple(T, W), triple(T, L)
    s, bound = kpm_scale(model, E; δ_s, B_H, kernel)
    w = 1 / (6 * N^2)                                        # ν* |Γ*_j| / N²
    μ = zeros(P)
    points = zeros(Int, 3)
    for j in 1:3
        q = [stack.B[j] * SVector(n₁, n₂) / N for n₂ in 0:N-1 for n₁ in 0:N-1]
        filter!(qᵢ -> norm(qᵢ - nearest_dirac(stack, j, qᵢ)[1]) < Wⱼ[j], q)
        moments = zeros(length(q), P)
        Threads.@threads for i in eachindex(q)
            window = Window(stack, q[i], Wⱼ, Lⱼ)
            z = zero_label(window, j)
            z > 0 || throw(ErrorException("the zero label of layer $j is not in the window at q = $(q[i])"))
            H̃ = scaled(hamiltonian(model, window), s, bound)
            moments[i, :] = chebyshev_moments(H̃, collect(orbitals(z)), P)
        end
        μ .+= w .* vec(sum(moments; dims = 1))
        points[j] = length(q)
    end
    return (E = collect(E), ρ = kpm_density(μ, s, E; kernel), μ = μ, s = s, B_H = bound, kernel = kernel,
        resolution = kpm_resolution(s, P, kernel), mass = μ[1], points = points)
end

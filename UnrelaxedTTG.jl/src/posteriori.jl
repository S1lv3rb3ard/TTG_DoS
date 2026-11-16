# A posteriori bounds for the truncation at a zero label (manuscript,
# Lemma 4.4, Remark 4.5 and Lemma C.4).
#
# Notation of Section 4.3: X is the set of retained labels, O the labels
# outside X that are coupled to X, 𝖧_Y the restriction of 𝖧 = Ĥ^τ_q to ℓ²(Y),
# 𝔥_{YY′} the coupling block, R_z(T) = (z − T)⁻¹, and
# Z = {t ± iε : dist(t, Σ) ≤ η} the contour. Here X is the whole `Window`, so
# the bounds cover the W- and the L-truncation together (Lemma C.4 holds for
# every X; Remark 4.5). All matrices are dense: use windows of at most a few
# thousand labels.
#
# The suprema over Z are taken over finitely many points t. They are computed
# diagnostics, not certified bounds (Remark 4.5).

"""
    Boundary

The labels `labels` of O and the coupling block `𝔥` = 𝔥_{OX} (rows: orbitals of
O, columns: orbitals of X).
"""
struct Boundary{T<:Real}
    labels::Vector{Label}
    𝔥::SparseMatrixCSC{Complex{T},Int}
end

"""
    boundary(model, window; L = Inf) -> Boundary

The labels outside `window` that are coupled to it by Ĥ^τ_q, and 𝔥_{OX}. With a
finite `L` only outside labels in ℒ*(L) are kept; then the bounds compare
Ĥ_q^{(τ,W,L)} with the L-truncated operator on ℒ*(L) instead of Ĥ^τ_q.
"""
function boundary(model::Model{T}, window::Window{T}; L = Inf) where {T}
    Lⱼ = triple(T, L)
    index = Dict{Label,Int}()
    labels = Label[]
    I, J, V = Int[], Int[], Complex{T}[]
    for (s, x) in enumerate(window.labels)
        neighbours(model, window.q, x) do y, block
            haskey(window.index, y) && return
            label_size(model.stack, y) < Lⱼ[y.layer] || return
            o = get!(index, y) do
                push!(labels, y)
                length(labels)
            end
            for β in 1:2, α in 1:2                         # [𝖧]_{(y,β),(x,α)} = conj [𝖧]_{(x,α),(y,β)}
                push!(I, 2o - 2 + β); push!(J, 2s - 2 + α); push!(V, conj(block[α, β]))
            end
        end
    end
    return Boundary{T}(labels, sparse(I, J, V, 2length(labels), dimension(window)))
end

"""The contour points t of Z: `nt` equally spaced points of [min Σ − η, max Σ + η]."""
contour(Σ, η::Real, nt::Integer) = range(minimum(Σ) - η, maximum(Σ) + η; length = nt)

"""
    flux_bound(model, window, orbital; Σ, ε, η, nt = 201, outside = boundary(model, window)) -> (bound, flux)

The vector form of the a posteriori bound (Remark 4.5, Lemma C.4) for the
diagonal entry at `orbital` of X = `window`: with v_z = 𝔥_{OX} R_z(𝖧_X) δ₀,

    |[R_z(𝖧) − R_z(𝖧_X)]₀₀| ≤ ε⁻¹ ‖v_z‖₂ ‖v_z̄‖₂      (z = t ± iε),

and by the contour representation (B.3), for all E ∈ Σ,

    |[δ_ε(E − 𝖧)]₀₀ − [δ_ε(E − 𝖧_X)]₀₀| ≤ (√e/π) `flux` + (2√e/(πε)) e^{−η²/(2ε²)} = `bound`,

where `flux` = sup_{z ∈ Z} ε⁻¹ ‖v_z‖₂ ‖v_z̄‖₂ (sampled at `nt` points).
"""
function flux_bound(model::Model{T}, window::Window{T}, orbital::Integer; Σ, ε::Real, η::Real, nt::Integer = 201,
        outside::Boundary{T} = boundary(model, window)) where {T}
    decomposition = eigen(Hermitian(Matrix(hamiltonian(model, window))))
    E_X, V = decomposition.values, decomposition.vectors
    𝔥V = outside.𝔥 * V                                     # 𝔥_{OX} V
    w = conj.(view(V, orbital, :))                          # V* δ₀
    v(z) = norm(𝔥V * (w ./ (z .- E_X)))                     # ‖𝔥_{OX} R_z(𝖧_X) δ₀‖₂
    flux = maximum(t -> v(complex(t, ε)) * v(complex(t, -ε)), contour(Σ, η, nt)) / ε
    return sqrt(ℯ) / π * flux + tail(ε, η), flux
end

"""The contour tail (2√e/(πε)) e^{−η²/(2ε²)} of (B.3)."""
tail(ε::Real, η::Real) = 2 * sqrt(ℯ) / (π * ε) * exp(-η^2 / (2ε^2))

"""
    transfer_norm(model, window, r; Σ, ε, η, nt = 201, outside = boundary(model, window)) -> (𝔟, g₀)

The barrier transfer norm of eq. (4.19) for the core I_r = {x ∈ X : ‖ζ(x)‖₂ ≤ r}
and the barrier B = X ∖ I_r,

    𝔟_q(I_r) = sup_{z ∈ Z} ‖𝔥_{OI} + 𝔥_{OB} R_z(𝖧_B) 𝔥_{BI}‖₂,

sampled at `nt` points t, and the barrier gap g₀ = dist(Σ, σ(𝖧_B)).
[`posteriori_bound`](@ref) turns 𝔟 into the bound (4.22).
"""
function transfer_norm(model::Model{T}, window::Window{T}, r::Real; Σ, ε::Real, η::Real, nt::Integer = 201,
        outside::Boundary{T} = boundary(model, window)) where {T}
    core = [norm(ζ) ≤ r for ζ in window.ζ]
    any(core) || throw(ArgumentError("the core I_r is empty"))
    I_orbitals = [o for s in eachindex(core) if core[s] for o in orbitals(s)]
    B_orbitals = [o for s in eachindex(core) if !core[s] for o in orbitals(s)]
    H = hamiltonian(model, window)
    # ‖𝔥_{OI} + 𝔥_{OB} T‖₂ = ‖R_I + R_B T‖₂ for the triangular factor R of 𝔥_{OX} = QR.
    R = size(outside.𝔥, 1) > size(outside.𝔥, 2) ? Matrix(qr(Matrix(outside.𝔥)).R) : Matrix(outside.𝔥)
    R_I = R[:, I_orbitals]
    isempty(B_orbitals) && return opnorm(R_I), T(Inf)
    decomposition = eigen(Hermitian(Matrix(H[B_orbitals, B_orbitals])))
    E_B, V = decomposition.values, decomposition.vectors
    left = R[:, B_orbitals] * V                             # R_B V
    right = V' * Matrix(H[B_orbitals, I_orbitals])          # V* 𝔥_{BI}
    transfer(z) = opnorm(R_I + left * (right ./ (z .- E_B)))
    𝔟 = maximum(t -> max(transfer(complex(t, ε)), transfer(complex(t, -ε))), contour(Σ, η, nt))
    return 𝔟, spectral_distance(Σ, E_B)
end

"""dist(Σ, σ) for the interval [min Σ, max Σ] and a list of eigenvalues."""
spectral_distance(Σ, eigenvalues) = minimum(E -> max(minimum(Σ) - E, E - maximum(Σ), zero(E)), eigenvalues)

"""
    posteriori_bound(𝔟, ε, η)

The bound (4.22) of Lemma 4.4(a): (√e/π) 𝔟²/ε³ + (2√e/(πε)) e^{−η²/(2ε²)}.
"""
posteriori_bound(𝔟::Real, ε::Real, η::Real) = sqrt(ℯ) / π * 𝔟^2 / ε^3 + tail(ε, η)

"""
    escape_norm(model, window, W₀, orbital; Σ, ε, η, nt = 201) -> (𝔰, g)

Escape and gap of eq. (4.20) inside Y = X = `window` for the set
A = {x ∈ X : ‖ζ(x)‖₂ > W₀} (on X, ‖ζ(x)‖₂ is the Dirac distance d_𝒦(x), Lemma C.1):

    𝔰_q(A; X) = sup_{z ∈ Z} ‖𝔥_{CA} R_z(𝖧_A) δ₀‖₂,   C = X ∖ A,      g_q(A) = dist(Σ, σ(𝖧_A)).

`orbital` must belong to A. [`bracket_bound`](@ref) gives the bound (4.21) on
[δ_ε(E − 𝖧_X)]₀₀, the second term of (4.23) for a zero label in the window.
"""
function escape_norm(model::Model{T}, window::Window{T}, W₀::Real, orbital::Integer; Σ, ε::Real, η::Real,
        nt::Integer = 201) where {T}
    far = [norm(ζ) > W₀ for ζ in window.ζ]
    far[(orbital+1)÷2] || throw(ArgumentError("the start orbital must lie in A (‖ζ‖₂ > W₀)"))
    A_orbitals = [o for s in eachindex(far) if far[s] for o in orbitals(s)]
    C_orbitals = [o for s in eachindex(far) if !far[s] for o in orbitals(s)]
    H = hamiltonian(model, window)
    decomposition = eigen(Hermitian(Matrix(H[A_orbitals, A_orbitals])))
    E_A, V = decomposition.values, decomposition.vectors
    g = spectral_distance(Σ, E_A)
    isempty(C_orbitals) && return zero(T), g
    𝔥V = H[C_orbitals, A_orbitals] * V
    w = conj.(view(V, findfirst(==(orbital), A_orbitals), :))
    escape(z) = norm(𝔥V * (w ./ (z .- E_A)))
    𝔰 = maximum(t -> max(escape(complex(t, ε)), escape(complex(t, -ε))), contour(Σ, η, nt))
    return 𝔰, g
end

"""
    bracket_bound(𝔰, g, ε, η)

The bracket ℬ_q(A; Y) of eq. (4.21):
(√e/π) 𝔰²/ε + (2√e/(πε)) e^{−η²/(2ε²)} + (ε√(2π))⁻¹ e^{−g²/(2ε²)}.
"""
bracket_bound(𝔰::Real, g::Real, ε::Real, η::Real) =
    sqrt(ℯ) / π * 𝔰^2 / ε + tail(ε, η) + exp(-g^2 / (2ε^2)) / (ε * sqrt(2π))

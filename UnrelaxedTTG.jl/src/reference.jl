# Slow reference implementations written directly from the definitions. They
# share no tables with the fast code and exist to test it.

"""
    reference_bloch(model, t, k) -> 2×2 Matrix

Bloch matrix (2.4) of layer t as a sum over lattice vectors R of layer t,

    h̃^{α}_{β}(k) = ∑_R e^{−ik·(R + τ_α − τ_β)} h(‖R + τ_α − τ_β‖₂, α, β),

where the hopping is looked up by distance and sublattice pair, minus E_D on
the diagonal.
"""
function reference_bloch(model::Model{T}, t::Integer, k) where {T}
    stack, shells = model.stack, model.shells
    amplitude(d, same) = begin
        radii, t_n = same ? (shells.dA, shells.tA) : (shells.dB, shells.tB)
        i = findfirst(r -> abs(r - d) < 1e-8 * stack.a, radii)
        i === nothing ? zero(T) : t_n[i]
    end
    h = zeros(Complex{T}, 2, 2)
    for m₂ in -8:8, m₁ in -8:8, α in 1:2, β in 1:2
        r = stack.A[t] * SVector(m₁, m₂) + orbital_offset(stack, t, α) - orbital_offset(stack, t, β)
        h[α, β] += cis(-(Vec2{T}(k) ⋅ r)) * amplitude(norm(r), α == β)
    end
    return h - model.E_D * I
end

"""
    reference_fourier_hopping(model, j, k, ξ; R = 8a, nᵣ = 400, n_φ = 720) -> 2×2 Matrix

ĥ_{kβ}^{jα}(ξ) = (2π)⁻² ∫ h_{kβ}^{jα}(x) e^{−iξ·x} dx (eq. (2.10)) by a
two-dimensional polar quadrature of the real-space hopping (F.5), without
Hankel transforms.
"""
function reference_fourier_hopping(model::Model{T}, j::Integer, k::Integer, ξ; R::Real = 8 * model.stack.a,
        nᵣ::Integer = 400, n_φ::Integer = 720, parameters::InterlayerParameters = FANG_KAXIRAS) where {T}
    stack = model.stack
    x, w = polar_quadrature(nᵣ, n_φ, R)
    ĥ = zeros(Complex{T}, 2, 2)
    for α in 1:2, β in 1:2
        ĥ[α, β] = sum(i -> w[i] * interlayer_hopping(x[i], parameters, stack.a, stack.τ̂[j], stack.τ̂[k], α, β) *
                           cis(-(Vec2{T}(ξ) ⋅ x[i])), eachindex(x)) / (2π)^2
    end
    return ĥ
end

"""
    reference_hamiltonian(model, window; quadrature = (;)) -> Matrix

Dense Ĥ_q^{(τ,W,L)} from Definition 3.3, entry by entry over all pairs of
labels of the window: for x = (G, jα), y = (G′, kβ),

    [Ĥ]_{xy} = δ_jk δ_GG′ h̃_{jβ}^{jα}(k_x)
             + (1 − δ_jk) δ_{G_l G′_l} c_j c_k g_τ(p) e^{i(G′_j·τ_jα − G_k·τ_kβ)} ĥ_{kβ}^{jα}(p),

p = q + G′_j + G_k + G_l, for the coupled pairs of layers, with
[`reference_bloch`](@ref) and [`reference_fourier_hopping`](@ref). For windows
of a few dozen labels.
"""
function reference_hamiltonian(model::Model{T}, window::Window{T}; quadrature = (;)) where {T}
    stack, q = model.stack, window.q
    H = zeros(Complex{T}, dimension(window), dimension(window))
    for (s, x) in enumerate(window.labels), (s′, y) in enumerate(window.labels)
        j, k = x.layer, y.layer
        rows, columns = (2s-1):2s, (2s′-1):2s′
        if s == s′
            H[rows, columns] = reference_bloch(model, j, momentum(stack, q, x))
        elseif j != k && k in coupled(j)
            l = third(j, k)
            x.m[l] == y.m[l] || continue
            G′ⱼ, Gₖ = stack.B[j] * y.m[j], stack.B[k] * x.m[k]
            p = q + G′ⱼ + Gₖ + stack.B[l] * x.m[l]
            g = momentum_cutoff(norm(p), model.τ, model.δ)
            g > 0 || continue
            ĥ = reference_fourier_hopping(model, j, k, p; quadrature...)
            for α in 1:2, β in 1:2
                phase = cis(G′ⱼ ⋅ orbital_offset(stack, j, α) - Gₖ ⋅ orbital_offset(stack, k, β))
                H[rows[α], columns[β]] = stack.Γ * g * exp(-(model.ς * norm(p))^2 / 2) * phase * ĥ[α, β]
            end
        end
    end
    return H
end

"""
    gaussian_density(H, orbital, E, ε) -> Vector

[δ_ε(E − H)]_{cc} for the orbital c = `orbital` of a Hermitian matrix, from the
eigendecomposition: ∑_λ |v_λ(c)|² δ_ε(E − λ), with the Gaussian δ_ε of Section 2.5.
"""
function gaussian_density(H::AbstractMatrix, orbital::Integer, E, ε::Real)
    decomposition = eigen(Hermitian(Matrix(H)))
    weight = abs2.(view(decomposition.vectors, orbital, :))
    return [sum(i -> weight[i] * exp(-(Eᵢ - decomposition.values[i])^2 / (2ε^2)), eachindex(weight)) /
            (ε * sqrt(2π)) for Eᵢ in E]
end

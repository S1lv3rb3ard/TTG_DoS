using UnrelaxedTTG
using LinearAlgebra, Random, SparseArrays, StaticArrays, Test

const U = UnrelaxedTTG

# The GPU test set at the end runs only if CUDA.jl is in the active environment and a device is
# usable. CUDA.jl is not a test dependency, so `Pkg.test` skips it; to run it, start this file in
# an environment that has UnrelaxedTTG, CUDA and StaticArrays.
const HAS_CUDA = Base.find_package("CUDA") !== nothing
HAS_CUDA && @eval using CUDA

Random.seed!(20261111)
const stack = Stack(1.42 * √3, (-1.4, 0.0, 2.8))
const model = Model(stack; τ = 8.5, δ = 6.5)

"""All labels of ℒ*(L) (a finite set), as a `Window` built from an explicit list."""
function all_labels(stack, q, L)
    reach = ceil(Int, L / stack.σ)
    labels = Label[]
    range = [SVector(a, b) for a in -reach:reach for b in -reach:reach]
    for t in 1:3, x in range, y in range
        label = Label(t, x, y)
        U.label_size(stack, label) < L && push!(labels, label)
    end
    sort!(labels)
    near = ntuple(t -> nearest_dirac(stack, t, q), 3)
    return Window(stack, U.Vec2{Float64}(q), SVector(Inf, Inf, Inf), SVector(L, L, L) .* 1.0,
        SVector(near[1][1], near[2][1], near[3][1]), SVector(near[1][2], near[2][2], near[3][2]), labels)
end

@testset "UnrelaxedTTG" begin

@testset "geometry (Section 2)" begin
    for j in 1:3
        @test stack.B[j]' * stack.A[j] ≈ 2π * I
        @test abs(det(stack.B[j])) ≈ stack.Γ
        @test norm(stack.K[j]) ≈ 4π / (3 * 1.42 * √3) ≈ stack.Knorm
        @test stack.τ[j] ≈ stack.A[j] * SVector(1 / 3, 1 / 3)
        @test minimum(svdvals(Matrix(stack.B[j]))) ≈ stack.σ
    end
    @test others(1) == (2, 3) && others(2) == (1, 3) && others(3) == (1, 2)
    for (j, t) in ((1, 2), (2, 3), (1, 3))
        @test dirac_separation(stack, j, t) ≈ norm(stack.K[j] - stack.K[t])     # eq. (4.11)
    end
    for t in 1:3, _ in 1:20
        γ = stack.B[t] * SVector(rand(-3:3), rand(-3:3))
        ζ = 0.3 * stack.Knorm * rand() * normalize(randn(SVector{2,Float64}))
        K̃, valley = nearest_dirac(stack, t, stack.K[t] + γ + ζ)
        @test K̃ ≈ stack.K[t] + γ && valley == 1
        K̃, valley = nearest_dirac(stack, t, stack.K′[t] + γ + ζ)
        @test K̃ ≈ stack.K′[t] + γ && valley == -1
    end
    lower, upper = connectedness_range(stack, 2)
    β₁₂, β₂₃ = dirac_separation(stack, 2, 1), dirac_separation(stack, 2, 3)
    @test lower ≈ β₁₂ + √3 / 2 * β₂₃
    @test upper ≈ stack.Knorm / 2
end

@testset "quadrature and cutoff (eq. (4.4))" begin
    x, w = gauss_legendre(12)
    @test sum(w) ≈ 2
    @test sum(w .* x .^ 22) ≈ 2 / 23
    p, v = polar_quadrature(40, 64, 3.0)
    @test sum(v) ≈ π * 9
    @test sum(i -> v[i] * exp(-sum(abs2, p[i])), eachindex(p)) ≈ π * (1 - exp(-9))
    τ, δ = 8.5, 6.5
    @test momentum_cutoff(τ - δ, τ, δ) == 1 && momentum_cutoff(0.0, τ, δ) == 1
    @test momentum_cutoff(τ, τ, δ) == 0 && momentum_cutoff(9.0, τ, δ) == 0
    @test momentum_cutoff(τ - δ / 2, τ, δ) ≈ 1 / 2
    r = range(τ - δ, τ; length = 200)
    @test issorted(momentum_cutoff.(r, τ, δ); rev = true)
    @test all(i -> momentum_cutoff(r[i], τ, δ) + momentum_cutoff(2τ - δ - r[i], τ, δ) ≈ 1, eachindex(r))
end

@testset "intralayer Bloch matrix (eq. (2.4), Appendix F)" begin
    @test dirac_energy(Shells(1.42 * √3; table = :I)) ≈ 0.00274 atol = 1e-12
    @test dirac_energy(Shells(1.42 * √3; table = :II)) ≈ 0.00317 atol = 1e-12
    @test model.E_D ≈ 0.00274 atol = 1e-12
    for t in 1:3
        @test norm(bloch(model.shells, stack, t, stack.K[t]; E_D = model.E_D)) < 1e-13        # Dirac point
        @test norm(bloch(model.shells, stack, t, stack.K′[t]; E_D = model.E_D)) < 1e-13
        for _ in 1:5
            k = 4 * randn(SVector{2,Float64})
            h̃ = bloch(model.shells, stack, t, k; E_D = model.E_D)
            @test h̃ ≈ h̃'
            @test h̃ ≈ reference_bloch(model, t, k) atol = 1e-13
            # periodicity in the gauge of (2.4): h̃(k + G) = D* h̃(k) D, D = diag(1, e^{iG·τ_t})
            G = stack.B[t] * SVector(2, -1)
            D = Diagonal(SVector(1, cis(G ⋅ stack.τ[t])))
            @test bloch(model.shells, stack, t, k + G; E_D = model.E_D) ≈ D' * h̃ * D atol = 1e-12
            @test opnorm(Matrix(h̃)) ≤ intralayer_bound(model.shells; E_D = model.E_D)
        end
    end
    # Dirac cone: the eigenvalues of h̃(K + ζ) are ±v‖ζ‖ + O(‖ζ‖²), and v is the same in every direction.
    v = [eigvals(Hermitian(Matrix(bloch(model.shells, stack, 2, stack.K[2] + 1e-5 * SVector(cos(φ), sin(φ));
        E_D = model.E_D))))[2] / 1e-5 for φ in (0.0, 0.7, 2.1)]
    @test maximum(v) - minimum(v) < 1e-3 * v[1]
    @test 5 < v[1] < 7                                                    # ħv_F ≈ 5.5–6.5 eV Å for graphene
    @test @inferred(bloch(model.shells, stack, 1, stack.K[1]; E_D = model.E_D)) isa SMatrix{2,2,ComplexF64,4}
    φ₁, φ₂ = band_gap(model.shells, stack, 0.1), band_gap(model.shells, stack, 0.3)
    @test 0 < φ₁ < φ₂
end

@testset "interlayer hopping and its Fourier transform (eqs. (F.5)–(F.8), (2.10))" begin
    τ̂ = stack.τ̂
    for (j, k) in ((1, 2), (2, 3)), α in 1:2, β in 1:2, _ in 1:5
        x = 3 * randn(SVector{2,Float64})
        h = interlayer_hopping(x, FANG_KAXIRAS, stack.a, τ̂[j], τ̂[k], α, β)
        @test h ≈ interlayer_hopping(-x, FANG_KAXIRAS, stack.a, τ̂[k], τ̂[j], β, α)       # Assumption 2.6
        # the 120° rotations of the two lattices: for an untwisted pair h is invariant
        R₃ = rotation(120)
        @test interlayer_hopping(x, FANG_KAXIRAS, stack.a, τ̂[j], τ̂[j], α, β) ≈
              interlayer_hopping(R₃ * x, FANG_KAXIRAS, stack.a, τ̂[j], τ̂[j], α, β)
    end
    # AB stacking of an untwisted pair: the B atom of the lower layer sits under the A atom of the
    # upper one; h(0) = V₀(0) = t₀ for every sublattice pair.
    @test interlayer_hopping(zero(SVector{2,Float64}), FANG_KAXIRAS, stack.a, τ̂[1], τ̂[2], 1, 2) ≈ FANG_KAXIRAS.t₀
    for (j, k) in ((1, 2), (2, 3), (2, 1)), _ in 1:4
        ξ = 2.5 * randn(SVector{2,Float64})
        norm(ξ) < model.τ || continue
        ĥ = fourier_hopping(model.ĥ, ξ, τ̂[j], τ̂[k])
        @test ĥ ≈ reference_fourier_hopping(model, j, k, ξ) atol = 1e-12
        @test fourier_hopping(model.ĥ, ξ, τ̂[k], τ̂[j]) ≈ ĥ' atol = 1e-15                # Hermiticity of (3.5)
    end
    # Hankel table against direct quadrature
    s, w = gauss_legendre(800)
    r = @. 8 * stack.a * (s + 1) / 2
    for ρ in (0.0, 0.37, 1.7031, 5.2, 8.5)
        direct = sum(i -> 4 * stack.a * w[i] * r[i] * U.radial(FANG_KAXIRAS, r[i] / stack.a)[1] *
                          U.besselj(0, ρ * r[i]), eachindex(r))
        @test hankel(model.ĥ, ρ)[1] ≈ direct atol = 1e-13
    end
    @test_throws DomainError hankel(model.ĥ, 8.6)
    # Interlayer coupling at the Dirac point, w = |Γ*| ĥ(K): about 0.11 eV for these parameters
    # (Fang & Kaxiras 2016), equal for AA and AB up to the twist.
    w_K = stack.Γ * fourier_hopping(model.ĥ, stack.K[1], τ̂[1], τ̂[2])
    @test 0.10 < abs(w_K[1, 1]) < 0.12
    @test abs(abs(w_K[1, 2]) - abs(w_K[1, 1])) < 0.01
    @test @inferred(fourier_hopping(model.ĥ, stack.K[1], τ̂[1], τ̂[2])) isa SMatrix{2,2,ComplexF64,4}
    @test (@allocated fourier_hopping(model.ĥ, stack.K[1], τ̂[1], τ̂[2])) == 0
end

@testset "windows (eqs. (4.7), (4.8), (4.27); Theorem 4.3)" begin
    q = stack.K[2] + SVector(0.011, -0.007)
    W, L = (0.25, 0.3, 0.27), (9.0, 8.0, 10.0)
    window = Window(stack, q, W, L)
    @test issorted(window.labels) && allunique(window.labels)
    @test length(window) == sum(length, window.layers)
    # membership by brute force over a box of integer coordinates
    reach = ceil(Int, 10.0 / stack.σ)
    box = [SVector(a, b) for a in -reach:reach for b in -reach:reach]
    for t in 1:3
        k, l = others(t)
        expected = Label[]
        for x in box
            norm(stack.B[k] * x) < L[t] || continue
            for y in box
                norm(stack.B[l] * y) < L[t] || continue
                label = Label(t, x, y)
                norm(momentum(stack, q, label) - window.K̃[t] - stack.B[t] * (x + y)) < W[t] && push!(expected, label)
            end
        end
        @test sort!(expected) == window.labels[window.layers[t]]
    end
    for (s, x) in enumerate(window.labels)
        @test window.ζ[s] ≈ shifted_momentum(stack, q, window.K̃, x)
        @test window.index[x] == s
    end
    @test all(t -> zero_label(window, t) > 0, 1:3)
    far = Window(stack, stack.K[2] + SVector(0.2, 0.0), 0.15, 6.0)       # ‖q − K̃_t‖ ≥ W for every layer
    @test all(t -> zero_label(far, t) == 0, 1:3)
    @test length(far) > 0                                                 # the windows are still built
    @test_throws ArgumentError Window(stack, q, 0.9, 5.0)
    @test_throws ArgumentError Window(stack, q, 0.3, Inf)
    # Theorem 4.3(b): for W in the connectedness range the breadth-first search returns the window
    for t in 1:3, shift in (SVector(0.0, 0.0), SVector(0.03, 0.05), SVector(-0.1, 0.02))
        lower, upper = connectedness_range(stack, t)
        Wₜ = 1.05 * lower
        qₜ = stack.K[t] + shift
        found = search_window(stack, qₜ, Wₜ, t; L = 30.0)
        direct = Window(stack, qₜ, Wₜ, 30.0)
        @test found == direct.labels[direct.layers[t]]
        # the same after a shift of q by a reciprocal vector of layer t (the test is ℛ*_t-periodic)
        @test search_window(stack, qₜ + stack.B[t] * SVector(1, -2), Wₜ, t; L = 30.0) == found
    end
end

@testset "truncated Hamiltonian (Definition 3.3)" begin
    q = stack.K[2] + SVector(0.01, 0.02)
    window = Window(stack, q, 0.2, 3.5)
    H = @inferred hamiltonian(model, window)
    @test H == H'
    @test @inferred(Window(stack, q, 0.2, 3.5)) isa Window{Float64}
    @test @inferred(interlayer_row_sum(model, 2, q)) isa SVector{2,Float64}
    @test size(H, 1) == dimension(window) == 2length(window)
    Href = reference_hamiltonian(model, window; quadrature = (nᵣ = 200, n_φ = 360))
    @test norm(Matrix(H) - Href) < 1e-11
    @test norm(Href - Href') < 1e-12
    @test count(!iszero, Href) == count(!iszero, Matrix(H))
    # `hamiltonian` computes each interlayer block from its lower layer; the blocks computed
    # from the two ends agree (Hermiticity of (3.5))
    for (s, x) in enumerate(window.labels)
        U.neighbours(model, q, x) do y, block
            s′ = get(window.index, y, 0)
            s′ == 0 && return
            @test Matrix(H[collect(orbitals(s)), collect(orbitals(s′))]) ≈ block atol = 1e-14
        end
    end
    # `hamiltonian` looks a label up before it evaluates its block; assembling from `neighbours`,
    # which evaluates every candidate, gives the same matrix entry for entry
    Is, Js, Vs = Int[], Int[], ComplexF64[]
    for (s, x) in enumerate(window.labels)
        h̃ = bloch(model.shells, stack, x.layer, momentum(stack, q, x); E_D = model.E_D)
        for β in 1:2, α in 1:2
            push!(Is, 2s - 2 + α); push!(Js, 2s - 2 + β); push!(Vs, h̃[α, β])
        end
        U.neighbours(model, q, x; upper = true) do y, block
            s′ = get(window.index, y, 0)
            s′ == 0 && return
            for β in 1:2, α in 1:2
                push!(Is, 2s - 2 + α); push!(Js, 2s′ - 2 + β); push!(Vs, block[α, β])
                push!(Is, 2s′ - 2 + β); push!(Js, 2s - 2 + α); push!(Vs, conj(block[α, β]))
            end
        end
    end
    @test sparse(Is, Js, Vs, size(H)...) == H
    # the candidates of a label are the labels of its nonzero blocks and of the blocks that vanish
    x = window.labels[1]
    all_candidates, nonzero = Label[], Label[]
    U.candidates((y, b, p, G′ₐ, φ_b) -> push!(all_candidates, y), model, q, x)
    U.neighbours((y, block) -> push!(nonzero, y), model, q, x)
    @test allunique(all_candidates) && nonzero ⊆ all_candidates && !isempty(nonzero)
    # the diagonal blocks are the Bloch matrices at k_x
    for s in (1, length(window) ÷ 2, length(window))
        x = window.labels[s]
        @test Matrix(H[collect(orbitals(s)), collect(orbitals(s))]) ≈
              bloch(model.shells, stack, x.layer, momentum(stack, q, x); E_D = model.E_D)
    end
    # layers 1 and 3 are not coupled, and the spectator coordinate is conserved
    rows, columns, _ = findnz(H)
    for (r, c) in zip(rows, columns)
        x, y = window.labels[(r+1)÷2], window.labels[(c+1)÷2]
        x == y && continue
        @test (x.layer, y.layer) in ((1, 2), (2, 1), (2, 3), (3, 2))
        l = U.third(x.layer, y.layer)
        @test x.m[l] == y.m[l]
    end
    # time reversal: the window at −q is the negated window and Ĥ_{−q}[−x, −y] = conj Ĥ_q[x, y]
    reversed = Window(stack, -q, 0.2, 3.5)
    negate(x) = Label(x.layer, -x.m)
    @test reversed.labels == sort(negate.(window.labels))
    @test reversed.valley == -window.valley
    order = reduce(vcat, [collect(orbitals(reversed.index[negate(x)])) for x in window.labels])
    @test hamiltonian(model, reversed)[order, order] ≈ conj(H) atol = 1e-13
    # covariance (the shift identity behind Proposition 4.10): moving q by G_k = B_k m and
    # relabelling G_k → G_k − B_k m in the labels of layers j and l gives a unitarily equivalent
    # matrix, so the diagonal moments at x_m = ((B_k m, 0), jα) equal those of the zero label at
    # q + B_k m. The relabelled window is passed explicitly, so the identity is exact at finite L.
    let j = 1, k = 2, m = SVector(1, -1), P = 24
        wide = Window(stack, stack.K[1] + SVector(0.004, 0.006), 0.2, 6.0)
        x_m = Label(j, setindex(zero(SVector{3,SVector{2,Int}}), m, k))
        @test haskey(wide.index, x_m)
        q′ = wide.q + stack.B[k] * m
        moved = sort([x.layer == k ? x : Label(x.layer, setindex(x.m, x.m[k] - m, k)) for x in wide.labels])
        near = ntuple(t -> nearest_dirac(stack, t, q′), 3)
        shifted = Window(stack, q′, wide.W, wide.L, SVector(near[1][1], near[2][1], near[3][1]),
            SVector(near[1][2], near[2][2], near[3][2]), moved)
        @test maximum(norm, shifted.ζ) < 0.2 + 1e-12            # the relabelled set is the window at q′
        @test Set(norm.(shifted.ζ) .|> (z -> round(z; digits = 9))) == Set(norm.(wide.ζ) .|> (z -> round(z; digits = 9)))
        scale = 0.95 / spectral_bound(model)[1]
        μ = chebyshev_moments(scale * hamiltonian(model, wide), collect(orbitals(wide.index[x_m])), P)
        μ′ = chebyshev_moments(scale * hamiltonian(model, shifted), collect(orbitals(zero_label(shifted, j))), P)
        @test μ ≈ μ′ atol = 1e-12
    end
    # the regularization (2.13) damps the interlayer entries and leaves the intralayer blocks
    damped = hamiltonian(Model(stack; τ = 8.5, δ = 6.5, ς = 0.3), window)
    offdiagonal(A) = A - blockdiag((A[2s-1:2s, 2s-1:2s] for s in 1:length(window))...)
    @test norm(offdiagonal(damped)) < norm(offdiagonal(H))
    @test norm((damped - offdiagonal(damped)) - (H - offdiagonal(H))) == 0
    # spectral bounds
    B_H, 𝔍 = spectral_bound(model)
    big = hamiltonian(model, Window(stack, q, 0.45, 12.0))
    @test U.row_sum_norm(big) ≤ B_H
    @test opnorm(Matrix(H)) ≤ U.row_sum_norm(H) + 1e-12
    @test U.row_sum_norm(big) ≤ spectral_bound(model, 0.45)[1] ≤ B_H
    for a in 1:3, _ in 1:10
        @test maximum(interlayer_row_sum(model, a, 5 * randn(SVector{2,Float64}))) ≤ 𝔍
    end
end

@testset "kernel polynomial method (Section 4.5)" begin
    g = jackson(50)
    @test g[1] ≈ 1
    @test all(>(0), g) && issorted(g[2:end]; rev = true)
    # (4.34) is twice the usual Jackson factor for n ≥ 1
    usual(n, P) = ((P - n + 1) * cos(π * n / (P + 1)) + sin(π * n / (P + 1)) * cot(π / (P + 1))) / (P + 1)
    @test g ≈ [(n == 0 ? 1 : 2) * usual(n, 50) for n in 0:49]
    A = Hermitian(randn(ComplexF64, 60, 60))
    s = 0.95 / opnorm(A)
    Ã = sparse(s * A)
    for P in (1, 2, 3, 4, 7, 64, 65)
        @test chebyshev_moments(Ã, [1, 5, 9], P) ≈ reference_moments(Ã, [1, 5, 9], P) atol = 1e-11
    end
    # moments against T_n(λ) from the eigendecomposition
    decomposition = eigen(Hermitian(Matrix(Ã)))
    exact = [sum(abs2.(decomposition.vectors[7, :]) .* cos.(n .* acos.(decomposition.values))) for n in 0:39]
    @test chebyshev_moments(Ã, [7], 40) ≈ exact atol = 1e-11
    # normalisation: ∫ δ^P_KPM dE = 1 per start orbital, and positivity (Jackson kernel)
    P = 300
    μ = chebyshev_moments(Ã, [7], P)
    θ = [π * (i - 1 / 2) / 4000 for i in 1:4000]                       # Chebyshev–Gauss rule in x = sE
    ρ = kpm_density(μ, s, cos.(θ) ./ s)
    @test sum(ρ .* sin.(θ)) * π / 4000 / s ≈ 1 atol = 1e-10
    @test minimum(ρ) > -1e-12
    # against the Gaussian-broadened density at the nominal resolution ε_P = π/(sP)
    E = range(-0.6 / s, 0.6 / s; length = 41)
    ε = kpm_resolution(s, P)
    @test norm(kpm_density(μ, s, E) - gaussian_density(A, 7, E, ε)) < 0.1 * norm(gaussian_density(A, 7, E, ε))
    @test_throws DomainError kpm_density(μ, s, [1.01 / s])
    @test_throws ErrorException U.scaled(sparse(Matrix(A)), s, 0.1)
end

@testset "momentum LDoS (eq. (4.36), convention (4.10))" begin
    E = range(-0.4, 0.4; length = 81)
    q = [stack.K[2] + SVector(0.01, 0.02), stack.K[2] + SVector(0.3, 0.0), stack.K[1]]
    W, L, P = 0.2, 6.0, 256
    result = momentum_ldos(model, q, E; W, L, P)
    @test size(result.ldos) == (3, 81)
    @test result.present == [3, 0, 3]                 # ‖K₁ − K₃‖ = 0.125 < W
    @test all(iszero, result.ldos[2, :])              # no zero label in the window: zero by (4.10)
    @test all(>(0), result.ldos[1, :])
    # the same number from dense eigenvalues
    window = Window(stack, q[1], W, L)
    H = hamiltonian(model, window)
    start = [o for t in 1:3 for o in orbitals(zero_label(window, t))]
    μ = reference_moments(result.s * H, start, P)
    @test result.μ[1, :] ≈ μ atol = 1e-10
    @test result.ldos[1, :] ≈ kpm_density(μ, result.s, E) / 6
    dense = sum(o -> gaussian_density(H, o, E, result.resolution), start) / 6
    @test norm(result.ldos[1, :] - dense) < 0.1 * norm(dense)
    @test result.resolution ≈ π / (result.s * P)
    path = linecut(stack, 4)
    @test length(path.q) == 21 && path.q[1] ≈ stack.K[1] && path.q[end] ≈ stack.K[3]
    @test path.q[5] ≈ stack.K[2] && path.q[17] ≈ stack.K[2]
    # Γ₁₂ is at distance ‖K₂ − K₁‖ from both corners (centre of the moiré hexagon)
    @test norm(path.q[9] - stack.K[1]) ≈ norm(path.q[9] - stack.K[2]) ≈ dirac_separation(stack, 1, 2)
end

@testset "density of states (Proposition 4.10, eq. (4.39); eq. (4.1))" begin
    E = range(-0.3, 0.3; length = 61)
    W, L = 0.2, 13.0
    short = @test_logs (:warn, r"outside") moire_dos(model, E; W, L = 5.0, N = 1, P = 8)
    @test sum(short.missing) > 0                          # centre labels outside ℒ*(L) are reported
    coarse = moire_dos(model, E; W, L, N = 2, P = 64)
    fine = moire_dos(model, E; W, L, N = 6, P = 64)
    @test sum(fine.missing) == 0
    @test fine.mass ≈ fine.disk_mass rtol = 0.02          # Riemann sum of the area of the disks
    @test abs(fine.mass - fine.disk_mass) ≤ abs(coarse.mass - coarse.disk_mass) + 1e-12
    @test fine.disk_mass ≈ 4π * 3 * W^2 / (6 * stack.Γ)
    @test all(>(0), fine.ρ)
    @test sum(fine.centres) * 2 == round(Int, sum(g -> sum(fine.point_moments[g][:, 1]), 1:6))
    # one sampling point by hand: layer 1, valley K, n = (1, 0)
    M, k = moire_cell(stack, 1)
    @test k == 2 && M ≈ stack.B[2] - stack.B[1]
    r = M * SVector(1, 0) / 6
    window = Window(stack, stack.K[1] + r, W, L)
    start = Int[]
    for (s, x) in enumerate(window.labels)
        x.layer == 1 && iszero(x.m[3]) && norm(window.ζ[s]) < W && append!(start, orbitals(s))
    end
    μ = reference_moments(fine.s * hamiltonian(model, window), start, 64)
    @test fine.point_moments[1][2, :] ≈ μ atol = 1e-10
    @test_throws ArgumentError moire_dos(model, E; W = 0.1, L, N = 2, P = 8)      # √3 β_jk > W
    @test moire_cell(stack, 2; layer2 = 3)[2] == 3
    grid = brillouin_dos(model, E; W, L, N = 6, P = 64)
    @test grid.points == [2, 2, 2]                        # K_j and K′_j are grid points for 3 | N
    @test grid.mass ≈ 3 * 2 * 2 / (6 * 36)
    @test all(>(0), grid.ρ)
end

@testset "a posteriori bounds (Lemma 4.4, Remark 4.5, Lemma C.4)" begin
    # Finite test problem: the L-truncated operator on all of ℒ*(L), which is a finite matrix,
    # is the reference; X is the window of radius W inside it.
    q = stack.K[2] + SVector(0.01, 0.02)
    L, W = 3.5, 0.2
    full = all_labels(stack, q, L)
    window = Window(stack, q, W, L)
    @test issubset(window.labels, full.labels) && length(full) > length(window)
    H_full, H_X = hamiltonian(model, full), hamiltonian(model, window)
    inside = [full.index[x] for x in window.labels]
    X = reduce(vcat, [collect(orbitals(s)) for s in inside])
    @test H_full[X, X] ≈ H_X                                       # 𝖧_X is the restriction
    outside = boundary(model, window; L)
    @test all(y -> haskey(full.index, y) && !haskey(window.index, y), outside.labels)
    O = reduce(vcat, [collect(orbitals(full.index[y])) for y in outside.labels])
    @test H_full[O, X] ≈ outside.𝔥                                  # 𝔥_{OX}
    rest = setdiff(1:size(H_full, 1), X, O)
    @test norm(H_full[rest, X]) == 0                                # O holds every label coupled to X
    Σ, ε, η = (-0.2, 0.2), 0.08, 0.4
    E = range(Σ[1], Σ[2]; length = 41)
    for t in 1:3
        o_X = orbitals(zero_label(window, t))[1]
        o_full = orbitals(zero_label(full, t))[1]
        difference = maximum(abs, gaussian_density(H_full, o_full, E, ε) - gaussian_density(H_X, o_X, E, ε))
        bound, flux = flux_bound(model, window, o_X; Σ, ε, η, outside)
        @test difference ≤ bound
        # resolvent form of Lemma C.4 at one point of the contour
        z = complex(0.05, ε)
        Δ = inv(z * I - Matrix(H_full))[o_full, o_full] - inv(z * I - Matrix(H_X))[o_X, o_X]
        @test abs(Δ) ≤ flux * (1 + 1e-9)
    end
    𝔟, g₀ = transfer_norm(model, window, 0.05; Σ, ε, η, outside)
    @test 𝔟 > 0 && g₀ ≥ 0
    o_X, o_full = orbitals(zero_label(window, 2))[1], orbitals(zero_label(full, 2))[1]
    difference = maximum(abs, gaussian_density(H_full, o_full, E, ε) - gaussian_density(H_X, o_X, E, ε))
    @test difference ≤ posteriori_bound(𝔟, ε, η)                    # eq. (4.22)
    # with the whole window as core, 𝔟 = ‖𝔥_{OX}‖₂
    @test transfer_norm(model, window, 1.0; Σ, ε, η, outside)[1] ≈ opnorm(Matrix(outside.𝔥))
    # bracket (4.21): a start orbital far from the Dirac points has little weight in Σ
    W₀ = 0.12
    s_far = findfirst(s -> norm(window.ζ[s]) > 0.19, eachindex(window.labels))
    orbital = orbitals(s_far)[1]
    𝔰, g = escape_norm(model, window, W₀, orbital; Σ, ε, η)
    @test maximum(gaussian_density(H_X, orbital, E, ε)) ≤ bracket_bound(𝔰, g, ε, η)
    @test_throws ArgumentError escape_norm(model, window, W₀, orbitals(zero_label(window, 2))[1]; Σ, ε, η)
end

@testset "angle sets and output folders of the drivers" begin
    @test triple_parameters(1) == ((-1.4, 0.0, 2.8), 0.4742, 19.40)
    @test triple_parameters(4) == ((-1.39, 0.0, 2.8), 0.4744, 19.555)
    @test length(ANGLE_SETS) == 4
    @test_throws ArgumentError triple_parameters(0)
    @test_throws ArgumentError triple_parameters(5)
    # rule (5.5), L = W / min|θ_j| with the angle in radians, for the four sets
    for (θ, W, L) in triple_parameters.(1:4)
        @test L ≈ W / deg2rad(minimum(abs, filter(!iszero, collect(θ)))) rtol = 2e-3
    end
    mktempdir() do directory
        folder = joinpath(directory, "run")
        meta = "theta -1.4 0.0 2.8\nW 0.4742\nP 100\n"
        @test read(claim_directory(folder, meta), String) == meta
        # no moments yet: another description takes the folder over
        other = "theta -1.4 0.0 2.8\nW 0.4\nP 200\n"
        claim_directory(folder, other)
        @test read(joinpath(folder, "meta.txt"), String) == other
        # with moments: the same description is accepted (line endings are not compared), another is refused
        write(joinpath(folder, "mu_0001.bin"), zeros(200))
        claim_directory(folder, other)
        claim_directory(folder, replace(other, "\n" => "\r\n"))
        claim_directory(folder, other)
        refusal = try claim_directory(folder, meta); nothing catch e; e end
        @test refusal isa ArgumentError && occursin("P, W", refusal.msg)
        @test read(joinpath(folder, "meta.txt"), String) == other          # nothing was written
        # moments without meta.txt cannot be attributed
        rm(joinpath(folder, "meta.txt"))
        @test_throws ArgumentError claim_directory(folder, other)
    end
end

@testset "GPU recurrence (extension UnrelaxedTTGCUDAExt)" begin
    if HAS_CUDA && CUDA.functional()
        q = stack.K[2] + SVector(0.01, 0.02)
        W = 0.25
        window = Window(stack, q, W, 6.0)
        B_H, _ = spectral_bound(model, W)
        s = 0.95 / B_H
        H̃ = U.scaled(hamiltonian(model, window), s, B_H)
        start = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
        @test length(start) == 6
        # the upload is the matrix itself, also after the conversion to single precision
        A = sprand(ComplexF64, 40, 40, 0.2)
        @test SparseMatrixCSC(gpu_matrix(A)) == A
        @test SparseMatrixCSC(gpu_matrix(A, ComplexF32)) == SparseMatrixCSC{ComplexF32,Int32}(A)
        # the same moments as on the CPU, for even and odd P
        for P in (1, 2, 3, 4, 7, 200, 201)
            μ = chebyshev_moments(H̃, start, P)
            @test chebyshev_moments(gpu_matrix(H̃), start, P) ≈ μ atol = 1e-11
            @test chebyshev_moments(gpu_matrix(H̃, ComplexF32), start, P) ≈ μ atol = 1e-4
        end
        @test chebyshev_moments(gpu_matrix(H̃), Int[], 5) == zeros(5)
    else
        @info "GPU test set skipped: CUDA.jl is not in the environment, or no device is usable"
    end
end

end

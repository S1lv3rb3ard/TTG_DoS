# Comparison of the two interlayer models, Fang–Kaxiras and the analytic refit.
#
#   using UnrelaxedTTG; include("examples/compare_interlayer.jl")
#   compare_models(; set = 2, W = 0.3771, L = 45.84)                 # transforms and Hamiltonians (CPU)
#   compare_linecuts(dir_FK, dir_R; output = "ldos_difference.txt")  # two line cuts of linecut_gpu
#
# For `compare_linecuts`, run `linecut_gpu` twice with the same set, W, L, P, Nq, precision and drop:
# first with `interlayer = :fang_kaxiras` (the default is the refit of orders (6, 7) since version
# 0.6.0), then with the refit (the default, or `interlayer = :analytic_refit` for orders (2, 3)) and
# `B_H` set to the B_H of the first run (from its meta.txt, all digits), so that both runs have the
# same scale s and resolution ε_P. Two folders of `production_linecut.jl` (CPU) work the same way, with the
# argument `B_H=<eV>` for the second run; a CPU folder and a GPU folder are not compared (their meta.txt
# differ in precision and drop_eV).

using UnrelaxedTTG, LinearAlgebra, Printf, SparseArrays

"""
    operator_norm(D; iterations = 300) -> Float64

Largest singular value of the sparse matrix `D` by the power method on D'D (a lower bound that
converges from below). If the relative change of the last step still exceeds 10⁻⁹ after
`iterations` steps, a warning gives it.
"""
function operator_norm(D; iterations::Int = 300)
    x = normalize(randn(ComplexF64, size(D, 2)))
    σ = 0.0
    for _ in 1:iterations
        y = D' * (D * x)
        σ_new = sqrt(norm(y))
        x = y / norm(y)
        change = abs(σ_new - σ) / σ_new
        σ = σ_new
        change ≤ 1e-9 && return σ
    end
    @warn "operator_norm: no convergence after $iterations steps" σ
    return σ
end

"""
    compare_models(; set, W, L, momenta = 1:60:301, Nq = 60, log = stdout, refit = ANALYTIC_REFIT_6_7, ς = 0) -> named tuple

The second model is `refit` (default: the default model, the refit of orders (6, 7)) with the Gaussian regularization (2.13) of length `ς` (e.g. `refit = FANG_KAXIRAS, ς = 0.01`
for the regularized Fang–Kaxiras model; the transforms below do not include the factor e^{−ς²ρ²/2}).
For the angle triple `set`: the Hankel transforms ℋ₃, ℋ₆ of Fang–Kaxiras and of `refit` on [0, τ] (largest
difference), the coupling Γ*ĥ(K₁) between layers 1 and 2, and at the momenta `momenta` of the line
cut (Nq steps per segment) the operator norm of Ĥ_q(refit) − Ĥ_q(Fang–Kaxiras) on the window
(W, L), with the operator norm of the interlayer part of Ĥ_q(Fang–Kaxiras) for scale.
"""
function compare_models(; set::Int, W::Real, L::Real, momenta = 1:60:301, Nq::Int = 60, log::IO = stdout,
                        refit::InterlayerModel = ANALYTIC_REFIT_6_7, ς::Real = 0)
    θ = triple_parameters(set)[1]
    stack = Stack(1.42 * √3, θ)
    FK = Model(stack; τ = 8.5, δ = 6.5, parameters = FANG_KAXIRAS)
    R = Model(stack; τ = 8.5, δ = 6.5, parameters = refit, ς)
    ρ = range(0, 8.5; length = 8501)
    ℋ_FK = hankel.(Ref(FK.ĥ), ρ)
    ℋ_R = hankel.(Ref(R.ĥ), ρ)
    d₃ = maximum(i -> abs(ℋ_R[i][2] - ℋ_FK[i][2]), eachindex(ρ))
    d₆ = maximum(i -> abs(ℋ_R[i][3] - ℋ_FK[i][3]), eachindex(ρ))
    @printf(log, "θ = %s: max_ρ≤τ |Δℋ₃| = %.3e, |Δℋ₆| = %.3e eV Å² (Γ*/(2π) times these: %.3f, %.3f meV)\n",
        string(θ), d₃, d₆, 1e3 * stack.Γ * d₃ / 2π, 1e3 * stack.Γ * d₆ / 2π)
    w_FK = stack.Γ * fourier_hopping(FK.ĥ, stack.K[1], stack.τ̂[1], stack.τ̂[2])
    w_R = stack.Γ * fourier_hopping(R.ĥ, stack.K[1], stack.τ̂[1], stack.τ̂[2])
    @printf(log, "coupling Γ*ĥ(K₁), layers 1-2, entry (A,A): %.4f meV (FK), %.4f meV (refit); (A,B): %.4f, %.4f meV\n",
        1e3 * abs(w_FK[1, 1]), 1e3 * abs(w_R[1, 1]), 1e3 * abs(w_FK[1, 2]), 1e3 * abs(w_R[1, 2]))
    B_FK, _ = spectral_bound(FK, W)
    B_R, _ = spectral_bound(R, W)
    @printf(log, "B_H(W = %.4f): %.6f eV (FK), %.6f eV (refit)\n", W, B_FK, B_R)
    path = linecut(stack, Nq)
    norms = Float64[]
    scales = Float64[]
    for i in momenta
        window = Window(stack, path.q[i], W, L)
        H_FK = hamiltonian(FK, window)
        D = hamiltonian(R, window) - H_FK
        inter = copy(H_FK)                                   # interlayer part: zero the same-label 2×2 blocks
        for s in 1:length(window)
            o = (2s - 1):(2s)
            inter[o, o] .= 0
        end
        push!(norms, operator_norm(D)); push!(scales, operator_norm(dropzeros!(inter)))
        @printf(log, "q %4d: %d labels, ‖ΔĤ_q‖ = %.4f meV, ‖Ĥ_q,inter(FK)‖ = %.4f eV\n", i, length(window),
            1e3 * norms[end], scales[end]); flush(log)
    end
    return (; θ, d₃, d₆, w_FK, w_R, B_FK, B_R, momenta, norms, scales)
end

read_meta(directory) = Dict(split(line, ' '; limit = 2)[1] => (occursin(' ', line) ? split(line, ' '; limit = 2)[2] : "")
                            for line in eachline(joinpath(directory, "meta.txt")))

"""
    compare_linecuts(dir_FK, dir_R; E = range(-0.2, 0.2; length = 1001), output = nothing, log = stdout)

Momentum LDoS F(q, E) (Jackson kernel, P moments) of two line cuts with the same parameters
except the interlayer model and with the same scale s; per momentum the relative difference
max_E |F_R − F_FK| / max_E |F_FK| (5.2) and the largest absolute difference (1/eV). Writes a table
to `output` if given: index, arc length, relative difference, absolute difference. Returns
(; relative, absolute, arclength, F_FK, F_R).
"""
function compare_linecuts(dir_FK::AbstractString, dir_R::AbstractString; E = range(-0.2, 0.2; length = 1001),
                          output = nothing, log::IO = stdout)
    m_FK, m_R = read_meta(dir_FK), read_meta(dir_R)
    get(m_FK, "interlayer", "fang_kaxiras") == "fang_kaxiras" || throw(ArgumentError("$dir_FK is not a Fang–Kaxiras run"))
    get(m_R, "interlayer", "fang_kaxiras") == "fang_kaxiras" && !haskey(m_R, "varsigma") &&
        throw(ArgumentError("$dir_R is a run of the unregularized Fang–Kaxiras model"))
    for key in ("theta", "W", "L", "P", "Nq", "B_H", "s", "precision", "drop_eV", "tau", "delta")
        # precision and drop_eV are absent from the meta.txt of the CPU driver production_linecut.jl
        get(m_FK, key, nothing) == get(m_R, key, nothing) ||
            throw(ArgumentError("the runs differ in $key: $(get(m_FK, key, "(none)")) and $(get(m_R, key, "(none)"))"))
    end
    P = parse(Int, m_FK["P"]); s = parse(Float64, m_FK["s"])
    ς_FK, ς_R = get(m_FK, "varsigma", "0"), get(m_R, "varsigma", "0")      # the regularization (2.13), Å
    @printf(log, "regularization length ς: %s Å (%s), %s Å (%s)\n", ς_FK, dir_FK, ς_R, dir_R)
    arclength = parse.(Float64, split(m_FK["arclength"]))
    n = length(arclength)
    F_FK = fill(NaN, n, length(E)); F_R = fill(NaN, n, length(E))
    relative = fill(NaN, n); absolute = fill(NaN, n)
    μ = Vector{Float64}(undef, P)
    for i in 1:n
        a, b = (joinpath(d, @sprintf("mu_%04d.bin", i)) for d in (dir_FK, dir_R))
        moments_complete(a, P) && moments_complete(b, P) || continue
        read!(a, μ); F_FK[i, :] = kpm_density(μ, s, E) ./ 6
        read!(b, μ); F_R[i, :] = kpm_density(μ, s, E) ./ 6
        absolute[i] = maximum(abs, F_R[i, :] .- F_FK[i, :])
        relative[i] = absolute[i] / maximum(abs, F_FK[i, :])
    end
    done = .!isnan.(relative)
    @printf(log, "%d of %d momenta in both runs; ε_P = %.3f meV; relative difference (5.2): median %.2e, max %.2e (q %d); max |ΔF| = %.3f 1/eV\n",
        count(done), n, 1e3 * kpm_resolution(s, P), sorted_median(relative[done]), maximum(relative[done]),
        findmax(x -> isnan(x) ? -Inf : x, relative)[2], maximum(absolute[done]))
    if output !== nothing
        open(output, "w") do io
            println(io, "# Fang–Kaxiras: ", dir_FK, ", ς = ", ς_FK, " Å"); println(io, "# refit: ", dir_R, ", ς = ", ς_R, " Å")
            println(io, "# index, arc length (1/Å), relative difference (5.2), max |F_R − F_FK| (1/eV)")
            for i in findall(done)
                @printf(io, "%d %.6f %.6e %.6e\n", i, arclength[i], relative[i], absolute[i])
            end
        end
    end
    return (; relative, absolute, arclength, F_FK, F_R, E)
end

sorted_median(x) = (y = sort(x); isodd(length(y)) ? y[(end+1)÷2] : (y[end÷2] + y[end÷2+1]) / 2)

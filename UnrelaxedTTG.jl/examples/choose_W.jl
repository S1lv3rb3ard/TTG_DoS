# Choice of the window radius W for a line cut from a short GPU sweep (measured differences).
#
#   using CUDA; include("examples/choose_W.jl")
#   result = choose_W(; set = 2, L = 45.84, P = 4000, candidates = (0.20, 0.25, 0.30), W_ref = 0.3771)
#
# At a few momenta of the line cut the momentum LDoS F_W(q, ·) (the average of (4.36) over the
# six zero-label orbitals) is computed for every candidate W and for the reference W_ref, all at
# the same L, P, arithmetic and threshold, and at the common KPM scale s = (1 − δ_s)/B_H with B_H
# from W_ref (so the Jackson kernel, and ε_P, are the same for every W). The W-difference of a
# candidate is the largest over the sampled momenta of the relative difference (5.2) to W_ref on
# Σ = [−0.2, 0.2] eV.
#
# The floor is measured in the same call: at the momenta `floor_momenta`, the difference (5.2)
# between W_ref at the production arithmetic and threshold and W_ref in double precision with
# the full matrix. The returned choice is the smallest candidate W such that it and every larger
# candidate differ from W_ref by no more than this floor, that is, the smallest W for which the
# W-truncation is not the largest of the measured differences. No tolerance is prescribed; if no
# candidate qualifies, `W_ref` is returned. The L- and P-dependence are not part of the sweep.

using UnrelaxedTTG, CUDA, Printf, LinearAlgebra, SparseArrays

# `triple_parameters(set)` (θ, default W and L of the angle triples) is exported by UnrelaxedTTG; `linecut_gpu` is not
# used here, so this file can be included on its own.

"""Relative difference (5.2): max_Σ |F − F_ref| / max_Σ |F_ref|."""
relative_difference(F, F_ref) = maximum(abs, F .- F_ref) / maximum(abs, F_ref)

"""
    build(model, q, W, L) -> (H, present, labels)

The sparse Ĥ_q^{(τ,W,L)} (full matrix), the zero-label orbitals present in the window and the
number of labels; zero labels outside the window contribute zero (convention (4.10)).
"""
function build(model, q, W, L)
    window = Window(model.stack, q, W, L)
    present = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
    return hamiltonian(model, window), present, length(window)
end

"""
    gpu_ldos(H, present, s, B_H, P; precision, drop, E) -> (F, seconds, μ)

F(E) = (1/6) ∑ over the zero-label orbitals of [δ^P_KPM(E, H)]_cc on the energies E, from moments
computed on the GPU at the scale s after removing the entries with |h| ≤ drop (eV). `seconds` is
the time of the upload and the recurrence.
"""
function gpu_ldos(H, present, s, B_H, P; precision = ComplexF32, drop = 1e-5, E)
    drop > 0 && (H = droptol!(copy(H), drop))
    S = SlicedELL(UnrelaxedTTG.scaled(H, s, B_H), precision; reach = present)   # rows reachable from the zero labels
    t = time()
    H̃ = gpu_matrix(S)
    μ = chebyshev_moments(H̃, present, P)
    CUDA.unsafe_free!(H̃)
    seconds = time() - t
    return kpm_density(μ, s, E) ./ 6, seconds, μ
end

"""The matrices of one momentum for several (W, L), assembled on parallel CPU threads."""
build_all(model, q, configurations) = fetch.([Threads.@spawn build(model, q, W, L) for (W, L) in configurations])

"""
    choose_W(; set, L, P, candidates, W_ref, Nq, momenta, floor_momenta, precision, drop, E, log, directory) -> result

The measured choice of the window radius W described at the top of this file. At the momenta
number `momenta` of the path `linecut(stack, Nq)` it computes, at the same `L`, `P`, `precision`
and `drop` and at the common scale s = 0.95/B_H with B_H = `spectral_bound(model, W_ref)[1]`,

    F_W(q, E) = (1/6) ∑_c [δ^P_KPM(E, Ĥ_q^{(τ,W,L)})]_cc      (momentum LDoS (4.36))

for every W in `candidates` and for `W_ref`, and the differences (5.2)

    d(W, q) = max_{E ∈ Σ} |F_W(q, E) − F_{W_ref}(q, E)| / max_{E ∈ Σ} |F_{W_ref}(q, E)|,      Σ = `E`.

The floor is the largest, over `floor_momenta`, of the same difference between F_{W_ref} in the
production arithmetic and F_{W_ref} in `ComplexF64` with the full matrix. The choice is the
smallest candidate W such that max_q d(W′, q) ≤ floor for it and for every larger candidate W′;
if there is none, `W_ref`.

Keyword arguments: `set` (angle triple, `triple_parameters`); `L`, `P`; `candidates` (each smaller
than `W_ref`); `W_ref`; `Nq` (default 60); `momenta` (default every 30th point); `floor_momenta`
(default the first and the middle one of `momenta`); `precision` (default `ComplexF32`); `drop`
(eV, default 10⁻⁵); `E` (default 4001 points of [−0.2, 0.2] eV); `log`; `directory` (if given, the
moments `mu_W<W>_q<index>.bin` and a `meta.txt` are written there).

`result` is a named tuple: `choice`, `floor`, `floor_values` (per momentum), `Ws` (sorted
candidates), `W_ref`, `momenta`, `difference` (candidates × momenta), `labels` and `seconds`
(candidates and, in the last row, W_ref × momenta), `B_H`, `s`. Pass `B_H` at full precision to
`linecut_gpu` to run the line cut at the same scale.
"""
function choose_W(; set::Int, L::Real, P::Int, candidates, W_ref::Real, Nq::Int = 60,
                  momenta = 1:30:(5Nq + 1), floor_momenta = (first(momenta), momenta[cld(length(momenta), 2)]),
                  precision::Type = ComplexF32, drop::Real = 1e-5,
                  E = range(-0.2, 0.2; length = 4001), log::IO = stdout, directory = nothing)
    θ = triple_parameters(set)[1]
    stack = Stack(1.42 * √3, θ)
    model = Model(stack; τ = 8.5, δ = 6.5)
    path = linecut(stack, Nq)
    B_H, _ = spectral_bound(model, W_ref)
    s = 0.95 / B_H
    Ws = sort(collect(Float64, candidates))
    all(<(W_ref), Ws) || throw(ArgumentError("every candidate must be smaller than W_ref"))
    @printf(log, "θ = %s, L = %.2f, P = %d, B_H = %.4f eV (from W_ref = %.4f), ε_P = %.3f meV, %s, drop %.0e eV\n",
        string(θ), L, P, B_H, W_ref, 1e3 * kpm_resolution(s, P), precision, drop); flush(log)
    difference = zeros(length(Ws), length(momenta))
    labels = zeros(Int, length(Ws) + 1, length(momenta))
    seconds = zeros(length(Ws) + 1, length(momenta))
    moments = directory === nothing ? nothing : Dict{Tuple{Float64,Int},Vector{Float64}}()
    floor_values = Dict{Int,Float64}()
    all(in(momenta), floor_momenta) || throw(ArgumentError("floor_momenta must be among momenta"))
    for (column, i) in enumerate(momenta)
        built = build_all(model, path.q[i], [(W, L) for W in [Ws; W_ref]])
        H, present, labels[end, column] = built[end]
        F_ref, seconds[end, column], μ = gpu_ldos(H, present, s, B_H, P; precision, drop, E)
        moments === nothing || (moments[(W_ref, i)] = μ)
        if i in floor_momenta
            F₆₄, = gpu_ldos(H, present, s, B_H, P; precision = ComplexF64, drop = 0.0, E)
            floor_values[i] = relative_difference(F_ref, F₆₄)
        end
        line = @sprintf("q %4d:", i)
        for (row, W) in enumerate(Ws)
            H, present, labels[row, column] = built[row]
            F, seconds[row, column], μ = gpu_ldos(H, present, s, B_H, P; precision, drop, E)
            moments === nothing || (moments[(W, i)] = μ)
            difference[row, column] = relative_difference(F, F_ref)
            line *= @sprintf(" W = %.3f: %.1e (%d labels, %.1f s);", W, difference[row, column], labels[row, column], seconds[row, column])
        end
        println(log, line * @sprintf(" W_ref: %d labels, %.1f s", labels[end, column], seconds[end, column])); flush(log)
    end
    for (i, d) in sort(collect(floor_values))
        @printf(log, "floor at q %4d (W_ref, %s with drop %.0e eV against ComplexF64, full matrix): %.1e\n", i, precision, drop, d)
    end
    floor = maximum(values(floor_values))
    worst = vec(maximum(difference; dims = 2))
    admissible = [all(worst[r:end] .≤ floor) for r in eachindex(Ws)]
    choice = any(admissible) ? Ws[findfirst(admissible)] : Float64(W_ref)
    for (row, W) in enumerate(Ws)
        @printf(log, "W = %.3f: difference to W_ref %.1e–%.1e, GPU %.1f s per momentum, %d labels\n", W,
            minimum(difference[row, :]), worst[row], sum(seconds[row, :]) / length(momenta), round(Int, sum(labels[row, :]) / length(momenta)))
    end
    @printf(log, "W_ref = %.4f: GPU %.1f s per momentum\nfloor %.1e; chosen W = %.3f\n", W_ref,
        sum(seconds[end, :]) / length(momenta), floor, choice); flush(log)
    if directory !== nothing
        mkpath(directory)
        for ((W, i), μ) in moments
            write(joinpath(directory, @sprintf("mu_W%.4f_q%04d.bin", W, i)), μ)
        end
        open(joinpath(directory, "meta.txt"), "w") do io
            println(io, "theta ", join(θ, " ")); println(io, "L ", L); println(io, "P ", P); println(io, "B_H ", B_H)
            println(io, "s ", s); println(io, "W_ref ", W_ref); println(io, "candidates ", join(Ws, " "))
            println(io, "momenta ", join(momenta, " ")); println(io, "precision ", precision); println(io, "drop_eV ", drop)
            println(io, "floor ", floor); println(io, "chosen_W ", choice)
        end
    end
    return (; choice, floor, floor_values, Ws, W_ref, momenta, difference, labels, seconds, B_H, s)
end

"""
    sweep_L(; set, W, Ls, L_ref, P, B_H, momenta, precision, drop, E, log) -> (; Ls, difference, labels, seconds)

The L-difference at fixed W: the relative difference (5.2) of F_L(q, ·) to F_{L_ref}(q, ·) at the
momenta `momenta` of the line cut, at the common scale s = 0.95/B_H (pass the B_H of the W-sweep).
"""
function sweep_L(; set::Int, W::Real, Ls, L_ref::Real, P::Int, B_H::Real, Nq::Int = 60, momenta = 1:60:(5Nq + 1),
                 precision::Type = ComplexF32, drop::Real = 1e-5, E = range(-0.2, 0.2; length = 4001), log::IO = stdout)
    θ = triple_parameters(set)[1]
    stack = Stack(1.42 * √3, θ)
    model = Model(stack; τ = 8.5, δ = 6.5)
    path = linecut(stack, Nq)
    s = 0.95 / B_H
    Ls = sort(collect(Float64, Ls))
    @printf(log, "θ = %s, W = %.4f, L_ref = %.3f, P = %d, B_H = %.4f eV, ε_P = %.3f meV, %s, drop %.0e eV\n",
        string(θ), W, L_ref, P, B_H, 1e3 * kpm_resolution(s, P), precision, drop); flush(log)
    difference = zeros(length(Ls), length(momenta))
    labels = zeros(Int, length(Ls) + 1, length(momenta))
    seconds = zeros(length(Ls) + 1, length(momenta))
    for (column, i) in enumerate(momenta)
        built = build_all(model, path.q[i], [(W, L) for L in [Ls; L_ref]])
        H, present, labels[end, column] = built[end]
        F_ref, seconds[end, column] = gpu_ldos(H, present, s, B_H, P; precision, drop, E)
        line = @sprintf("q %4d:", i)
        for (row, L) in enumerate(Ls)
            H, present, labels[row, column] = built[row]
            F, seconds[row, column] = gpu_ldos(H, present, s, B_H, P; precision, drop, E)
            difference[row, column] = relative_difference(F, F_ref)
            line *= @sprintf(" L = %.2f: %.1e (%d labels, %.1f s);", L, difference[row, column], labels[row, column], seconds[row, column])
        end
        println(log, line * @sprintf(" L_ref: %d labels, %.1f s", labels[end, column], seconds[end, column])); flush(log)
    end
    return (; Ls, L_ref, momenta, difference, labels, seconds)
end

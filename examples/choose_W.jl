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
#
# Per momentum only the reference matrix (W_ref, L) is assembled; every candidate is its principal
# submatrix on the labels of the smaller window (`restrict`, exact: the windows are nested and the
# entries do not depend on W or L). For the next `ahead` momenta (default 3) the reference matrix,
# the submatrices and their GPU format are prepared on CPU threads while the GPU works on the
# current momentum. `sweep_L` does the same with
# the reference (W, L_ref).

using UnrelaxedTTG, CUDA, Printf, LinearAlgebra, SparseArrays

# `triple_parameters(set)` (θ, default W and L of the angle triples) is exported by UnrelaxedTTG; `linecut_gpu` is not
# used here, so this file can be included on its own.
#
# The keyword `kernel` of `choose_W` and `sweep_L` selects the regularisation of the LDoS that is
# compared: `Jackson()` (default, as described above), `DeltaChebyshev(η, m)` (arXiv:2512.03149) or
# `Gaussian(ε)` (the Gaussian δ_ε expanded in Chebyshev polynomials).

include(joinpath(@__DIR__, "kernel_argument.jl"))            # kernel_text

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
    gpu_format(H, present, s, B_H; precision, drop) -> S

The matrix of the recurrence on the CPU side: the entries with |h| ≤ `drop` (eV) removed, scaled to
sH by `scaled` (no shift; it checks ‖H‖_∞ ≤ B_H), in the sliced ELLPACK format with the rows
reachable from the zero-label orbitals `present` (`SlicedELL`), in the arithmetic `precision`.
"""
gpu_format(H, present, s, B_H; precision = ComplexF32, drop = 1e-5) =
    SlicedELL(UnrelaxedTTG.scaled(drop > 0 ? droptol!(copy(H), drop) : H, s, B_H), precision; reach = present)

"""
    gpu_ldos(H, present, s, B_H, P; precision, drop, E, kernel = Jackson()) -> (F, seconds, μ)
    gpu_ldos(S::SlicedELL, present, s, P; E, kernel = Jackson()) -> (F, seconds, μ)

F(E) = (1/6) ∑ over the zero-label orbitals of [δ^P_KPM(E, H)]_cc (or of the kernel `kernel`) on the energies E, from moments
computed on the GPU at the scale s after removing the entries with |h| ≤ drop (eV). The second
form takes the matrix already in the GPU format (`gpu_format`). `seconds` is the time of the
upload and the recurrence.
"""
function gpu_ldos(S::SlicedELL, present, s, P; E, kernel = Jackson())
    t = time()
    H̃ = gpu_matrix(S)
    μ = chebyshev_moments(H̃, present, P)
    CUDA.unsafe_free!(H̃)
    seconds = time() - t
    return kpm_density(μ, s, E; kernel) ./ 6, seconds, μ
end
gpu_ldos(H, present, s, B_H, P; precision = ComplexF32, drop = 1e-5, E, kernel = Jackson()) =
    gpu_ldos(gpu_format(H, present, s, B_H; precision, drop), present, s, P; E, kernel)

zero_orbitals(window) = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]

"""
    prepared_ahead(prepare, items; ahead = 3) -> next

Runs `prepare(item)` for the items in order on CPU threads, up to `ahead` items in advance, so
that the CPU work (assembly of the reference matrix, the submatrices of the smaller windows and
their GPU format) overlaps with the GPU work on the current item. Each call `next()` returns the
result for the next item.
"""
function prepared_ahead(prepare, items; ahead::Int = 3)
    items = collect(items)
    tasks = Any[nothing for _ in items]
    spawn(c) = c ≤ length(items) && (tasks[c] = Threads.@spawn prepare(items[c]))
    foreach(spawn, 1:max(ahead, 1))
    count = Ref(0)
    return function next()
        c = count[] += 1
        result = fetch(tasks[c]); tasks[c] = nothing
        spawn(c + max(ahead, 1))
        return result
    end
end

"""
    reference_and_restrictions(model, q, W, L, smaller; s, B_H, precision, drop, keep) -> prepared

Assembles Ĥ_q at (W, L) once and returns, in the GPU format, it and its restrictions to the windows
(W′, L′) in `smaller` (`restrict`): `prepared.reference = (S, present, labels)`,
`prepared.smaller[k] = (S, present, labels)`, and `prepared.H` = the assembled matrix if `keep`.
"""
function reference_and_restrictions(model, q, W, L, smaller; s, B_H, precision, drop, keep::Bool = false)
    window = Window(model.stack, q, W, L)
    H = hamiltonian(model, window)
    present = zero_orbitals(window)
    reference = (gpu_format(H, present, s, B_H; precision, drop), present, length(window))
    restricted = map(smaller) do (W′, L′)
        H′, present′, labels′ = restrict(model.stack, window, H, W′, L′)
        (gpu_format(H′, present′, s, B_H; precision, drop), present′, labels′)
    end
    return (; reference, smaller = restricted, H = keep ? H : nothing, present)
end

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
(eV, default 10⁻⁵); `E` (default 4001 points of [−0.2, 0.2] eV); `log`; `directory` (if given, it is
locked for the call (`lock_directory`) and claimed with `claim_directory` before the sweep, so a
second run in it at the same time and a folder with moments of other parameters are refused; the parameters go to `meta.txt`, the moments of every W to `mu_W<W, six decimals>_q<index>.bin`
as soon as they are computed, and floor and choice to `result.txt`); `ahead` (default 3, momenta prepared in
advance on CPU threads).

`result` is a named tuple: `choice`, `floor`, `floor_values` (per momentum), `Ws` (sorted
candidates), `W_ref`, `momenta`, `difference` (candidates × momenta), `labels` and `seconds`
(candidates and, in the last row, W_ref × momenta), `B_H`, `s`. Pass `B_H` at full precision to
`linecut_gpu` to run the line cut at the same scale.
"""
function choose_W(; directory = nothing, kwargs...)
    directory === nothing && return run_choose_W(; kwargs...)
    lock_directory(directory)                            # one run per folder at a time
    try
        return run_choose_W(; directory, kwargs...)
    finally
        unlock_directory(directory)
    end
end

function run_choose_W(; set::Int, L::Real, P::Int, candidates, W_ref::Real, Nq::Int = 60,
                  momenta = 1:30:(5Nq + 1), floor_momenta = (first(momenta), momenta[cld(length(momenta), 2)]),
                  precision::Type = ComplexF32, drop::Real = 1e-5,
                  E = range(-0.2, 0.2; length = 4001), log::IO = stdout, directory = nothing, kernel = Jackson(),
                  interlayer::Symbol = DEFAULT_INTERLAYER, ahead::Int = 3)
    θ = triple_parameters(set)[1]
    stack = Stack(1.42 * √3, θ)
    model = Model(stack; τ = 8.5, δ = 6.5, interlayer)
    path = linecut(stack, Nq)
    B_H, _ = spectral_bound(model, W_ref)
    s = 0.95 / B_H
    Ws = sort(collect(Float64, candidates))
    all(<(W_ref), Ws) || throw(ArgumentError("every candidate must be smaller than W_ref"))
    @printf(log, "θ = %s, L = %.2f, P = %d, B_H = %.4f eV (from W_ref = %.4f), %s, %s, drop %.0e eV\n",
        string(θ), L, P, B_H, W_ref, kernel_text(kernel, s, P), precision, drop); flush(log)
    println(log, "interlayer model: ", interlayer)
    difference = zeros(length(Ws), length(momenta))
    labels = zeros(Int, length(Ws) + 1, length(momenta))
    seconds = zeros(length(Ws) + 1, length(momenta))
    if directory !== nothing
        meta = sprint() do io
            println(io, "theta ", join(θ, " ")); println(io, "L ", L); println(io, "P ", P); println(io, "B_H ", B_H)
            println(io, "s ", s); println(io, "W_ref ", W_ref); println(io, "candidates ", join(Ws, " "))
            println(io, "Nq ", Nq); println(io, "momenta ", join(momenta, " ")); println(io, "precision ", precision)
            println(io, "drop_eV ", drop); println(io, "tau 8.5"); println(io, "delta 6.5")
            interlayer_meta(io, interlayer)
        end
        claim_directory(directory, meta; lock = false)   # locked by choose_W; throws if the folder holds moments of other parameters
    end
    save(W, i, μ) = directory === nothing || write_moments(joinpath(directory, @sprintf("mu_W%.6f_q%04d.bin", W, i)), μ)
    floor_values = Dict{Int,Float64}()
    all(in(momenta), floor_momenta) || throw(ArgumentError("floor_momenta must be among momenta"))
    prepare(i) = reference_and_restrictions(model, path.q[i], W_ref, L, [(W, L) for W in Ws];
                                            s, B_H, precision, drop, keep = i in floor_momenta)
    next_prepared = prepared_ahead(prepare, momenta; ahead)   # Ĥ_q at (W_ref, L); the candidates are its submatrices
    for (column, i) in enumerate(momenta)
        prepared = next_prepared()
        S, present, labels[end, column] = prepared.reference
        F_ref, seconds[end, column], μ = gpu_ldos(S, present, s, P; E, kernel)
        save(W_ref, i, μ)
        if i in floor_momenta
            F₆₄, = gpu_ldos(prepared.H, present, s, B_H, P; precision = ComplexF64, drop = 0.0, E, kernel)
            floor_values[i] = relative_difference(F_ref, F₆₄)
        end
        line = @sprintf("q %4d:", i)
        for (row, W) in enumerate(Ws)
            S, present, labels[row, column] = prepared.smaller[row]
            F, seconds[row, column], μ = gpu_ldos(S, present, s, P; E, kernel)
            save(W, i, μ)
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
        open(joinpath(directory, "result.txt"), "w") do io
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
                 precision::Type = ComplexF32, drop::Real = 1e-5, E = range(-0.2, 0.2; length = 4001), log::IO = stdout,
                 kernel = Jackson(), interlayer::Symbol = DEFAULT_INTERLAYER, ahead::Int = 3)
    θ = triple_parameters(set)[1]
    stack = Stack(1.42 * √3, θ)
    model = Model(stack; τ = 8.5, δ = 6.5, interlayer)
    path = linecut(stack, Nq)
    s = 0.95 / B_H
    Ls = sort(collect(Float64, Ls))
    @printf(log, "θ = %s, W = %.4f, L_ref = %.3f, P = %d, B_H = %.4f eV, %s, %s, drop %.0e eV\n",
        string(θ), W, L_ref, P, B_H, kernel_text(kernel, s, P), precision, drop); flush(log)
    println(log, "interlayer model: ", interlayer)
    difference = zeros(length(Ls), length(momenta))
    labels = zeros(Int, length(Ls) + 1, length(momenta))
    seconds = zeros(length(Ls) + 1, length(momenta))
    prepare(i) = reference_and_restrictions(model, path.q[i], W, L_ref, [(W, L) for L in Ls]; s, B_H, precision, drop)
    next_prepared = prepared_ahead(prepare, momenta; ahead)   # Ĥ_q at (W, L_ref); the candidates are its submatrices
    for (column, i) in enumerate(momenta)
        prepared = next_prepared()
        S, present, labels[end, column] = prepared.reference
        F_ref, seconds[end, column] = gpu_ldos(S, present, s, P; E, kernel)
        line = @sprintf("q %4d:", i)
        for (row, L) in enumerate(Ls)
            S, present, labels[row, column] = prepared.smaller[row]
            F, seconds[row, column] = gpu_ldos(S, present, s, P; E, kernel)
            difference[row, column] = relative_difference(F, F_ref)
            line *= @sprintf(" L = %.2f: %.1e (%d labels, %.1f s);", L, difference[row, column], labels[row, column], seconds[row, column])
        end
        println(log, line * @sprintf(" L_ref: %d labels, %.1f s", labels[end, column], seconds[end, column])); flush(log)
    end
    return (; Ls, L_ref, momenta, difference, labels, seconds)
end

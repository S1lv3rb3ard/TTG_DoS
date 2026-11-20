# Production line cut on a GPU, with checkpoints. Same file format as `production_linecut.jl` (whose meta.txt
# has no precision and drop_eV lines, so a folder of one driver is not resumed by the other).
#
#   using CUDA; include("examples/production_linecut_gpu.jl")
#   linecut_gpu(; set = 2, P = 2000, Nq = 60, directory = "…")
#
# The Hamiltonians Ĥ_q^{(τ,W,L)} are assembled on the CPU threads (`builders` tasks), reduced to
# the rows that the recurrence can reach from the zero labels and stored in sliced ELLPACK form
# (`SlicedELL`), and passed through a bounded queue to `consumers` GPU tasks, each of which runs
# the Chebyshev recurrence for the six zero-label orbitals of one momentum with one kernel launch
# per step. Finished momenta (mu_<index>.bin) are skipped; a folder
# with moments of other parameters is refused, and the first error of a task stops the run.
# `precision = ComplexF32` runs the recurrence in single precision; `drop` (eV) removes the
# matrix entries with |h| ≤ drop before the recurrence (0 keeps the full matrix).
# `B_H` (eV) overrides the bound spectral_bound(model, W), e.g. to keep the KPM scale of the W-sweep of
# `choose_W.jl`. `cpu_workers` further tasks compute whole momenta on the CPU (same precision and matrix).

using UnrelaxedTTG, CUDA, Printf, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "kernel_argument.jl"))            # interlayer_meta

"""
    linecut_gpu(; set, P, Nq, directory, W, L, B_H, interlayer, ς, precision, drop, consumers, builders, queue, budget, log, cpu_workers) -> done

The Chebyshev moments of the momentum LDoS (4.36) along the path K₁–K₂–Γ₁₂–Γ₂₃–K₂–K₃
(`linecut(stack, Nq)`, 5`Nq` + 1 momenta), with the recurrence of Section 4.5 on the GPU. For every
momentum q of the path:

1. a CPU task builds the window 𝒲*(q, W) ∩ ℒ*(L) and the sparse Ĥ_q^{(τ,W,L)} (τ = 8.5, δ = 6.5 Å⁻¹,
   Jung–MacDonald Table I with the Dirac shift), removes the entries with |h| ≤ `drop`, checks
   ‖Ĥ_q‖_∞ ≤ B_H, scales by s = (1 − δ_s)/B_H, δ_s = 0.05, and stores the block of sĤ_q on the rows
   connected to the zero-label orbitals as a `SlicedELL` of element type `precision` (the other
   rows of the window cannot influence the moments, see `reachable`);
2. a GPU task computes μ_n = ∑_c [T_n(sĤ_q)]_cc, n = 0, …, P − 1, summed over the orbitals c of the
   zero labels present in the window (at most six), and writes them to `mu_<index>.bin`.

The LDoS is recovered afterwards as `kpm_density(μ, s, E) ./ 6` with the `s` of `meta.txt`
(the division by six, not by the number of orbitals present, is the convention (4.10)). The moments
do not depend on the regularisation kernel: `kpm_density(μ, s, E; kernel = DeltaChebyshev(η, m)) ./ 6`
evaluates the same files with the delta-Chebyshev kernel of arXiv:2512.03149, and
`kpm_density(μ, s, E; kernel = Gaussian(ε)) ./ 6` with the Gaussian δ_ε of width ε.

Keyword arguments:

- `set`: the angle triple, see [`triple_parameters`](@ref).
- `interlayer`: the interlayer model by name, `:analytic_refit_6_7` (default, the manuscript,
  `DEFAULT_INTERLAYER`), `:fang_kaxiras` or `:analytic_refit` (see `INTERLAYER_MODELS`), or a model
  object (e.g. an `AnalyticRefit` of higher order). A run with a refit gets the line
  `interlayer <name>` in its meta.txt (a Fang–Kaxiras run none, as before version 0.6.0),
  a run with a model object the line `interlayer <the object with all its parameters>`, so the
  moments of different models cannot be mixed.
- `ς`: length (Å) of the Gaussian regularization (2.13), ĥ → e^{−ς²‖ξ‖²/2} ĥ (default 0, none). A run
  with ς > 0 gets the line `varsigma <ς>` in its meta.txt.
- `P`: number of Chebyshev moments; the resolution is ε_P = π/(sP).
- `Nq`: steps per segment of the path.
- `directory`: output folder (created if needed).
- `W`, `L`: window radius (Å⁻¹) and scattering radius (Å⁻¹); default: the values of `triple_parameters(set)`.
- `B_H` (eV): bound used for the scale; default `spectral_bound(model, W)[1]`. Pass the B_H of a
  reference W (at full precision, from its `meta.txt`) to compare runs at the same s and ε_P.
- `precision`: element type of the recurrence, `ComplexF64` (default) or `ComplexF32`.
- `drop` (eV): entries with |h| ≤ `drop` are removed before the recurrence (0 keeps the full matrix).
  The change of Ĥ_q is bounded in norm by the largest removed row sum; it is not estimated here.
- `consumers`: number of GPU tasks (default 2, at least 1); `builders`: number of CPU assembly tasks
  (default: the number of Julia threads − `consumers` − 1, at least 1); `queue`: assembled matrices held between them.
- `cpu_workers`: further tasks that compute whole momenta on the CPU (default 0; measured on
  15 November 2026 to slow the run down, keep 0).
- `budget` (s): no new momentum is started after this time (default: no limit).
- `log`: stream for the progress lines (one per momentum).

Files in `directory`:

- `meta.txt`: one `key value…` line each for theta, W, L, P, Nq, B_H, J (= 𝔍), delta_s, s, precision,
  drop_eV, tau, delta, the intralayer table with E_D, interlayer (not for Fang–Kaxiras), varsigma
  (only for ς > 0), ticks and arclength (the arc lengths of the vertices and of all momenta, Å⁻¹).
- `mu_<index>.bin`: the `P` moments of momentum number `index` (1-based, four digits) as raw
  `Float64` in the byte order of the machine. Read with `read!(name, Vector{Float64}(undef, P))`.

Momenta whose file exists are skipped, so an interrupted run continues where it stopped. A folder
that holds moments is accepted only if its `meta.txt` equals that of the present call
([`claim_directory`](@ref)); otherwise an `ArgumentError` names the parameters that differ. Note
that `B_H` must then be passed with all its digits. Returns the number of momenta computed in
this call.

Errors. If a task fails (for example, the GPU runs out of memory, or ‖Ĥ_q‖_∞ > B_H), the error is
written to `log`, the other tasks stop after the momentum they are working on, and the call
throws the error. The moments written until then are kept, so the same call resumes the run.
If the call itself is interrupted (Ctrl-C), it first stops all its tasks and waits until they have
finished their current momentum, so that no task writes into the folder after the call has
returned. A file that is missing, empty or cut short is computed again (`moments_complete`).

The Julia session must have several threads (`julia -t auto`); `using CUDA` must come before
`include` of this file.
"""
function linecut_gpu(; set::Int, P::Int, Nq::Int, directory::String, W = nothing, L = nothing, B_H = nothing,
                     consumers::Int = 2, builders::Int = max(1, Threads.nthreads() - consumers - 1),
                     queue::Int = 6, budget::Real = Inf, log::IO = stdout, precision::Type = ComplexF64, drop::Real = 0.0, cpu_workers::Int = 0,
                     interlayer::Union{Symbol,InterlayerModel} = DEFAULT_INTERLAYER, ς::Real = 0)
    consumers ≥ 1 || throw(ArgumentError("consumers must be at least 1 (the GPU tasks); for a run on the CPU only use production_linecut.jl"))
    θ, W₀, L₀ = triple_parameters(set)
    W = something(W, W₀); L = something(L, L₀)
    stack = Stack(1.42 * √3, θ)
    model = interlayer isa Symbol ? Model(stack; τ = 8.5, δ = 6.5, ς, interlayer) :
                                    Model(stack; τ = 8.5, δ = 6.5, ς, parameters = interlayer)
    path = linecut(stack, Nq)
    δ_s = 0.05
    B_W, 𝔍 = spectral_bound(model, W)
    B_H = something(B_H, B_W)                          # a larger B_H (e.g. that of the reference W of choose_W) is allowed
    s = (1 - δ_s) / B_H
    meta = sprint() do io
        println(io, "theta ", join(θ, " ")); println(io, "W ", W); println(io, "L ", L); println(io, "P ", P)
        println(io, "Nq ", Nq); println(io, "B_H ", B_H); println(io, "J ", 𝔍); println(io, "delta_s ", δ_s); println(io, "s ", s)
        println(io, "precision ", precision); println(io, "drop_eV ", drop); println(io, "tau 8.5"); println(io, "delta 6.5"); println(io, "table I, Dirac shift E_D ", model.E_D)
        interlayer_meta(io, interlayer)
        ς > 0 && println(io, "varsigma ", ς)
        println(io, "ticks ", join(path.ticks, " ")); println(io, "arclength ", join(path.s, " "))
    end
    claim_directory(directory, meta)                   # throws if the folder holds moments of a run with other parameters
    file(i) = joinpath(directory, @sprintf("mu_%04d.bin", i))
    pending = [i for i in eachindex(path.q) if !moments_complete(file(i), P)]
    @printf(log, "%d of %d momenta to do; B_H = %.4f eV, ε_P = %.3f meV, %d builders, %d GPU tasks, interlayer %s\n", length(pending),
        length(path.q), B_H, 1e3 * kpm_resolution(s, P), builders, consumers, interlayer_label(interlayer) * (ς > 0 ? ", ς = $ς Å" : "")); flush(log)
    start = time()
    todo = Channel{Int}(length(pending) + 1); foreach(i -> put!(todo, i), pending); close(todo)
    built = Channel{Tuple{Int,SlicedELL{precision,Vector{precision},Vector{Int32}},Vector{Int},Int}}(queue)
    done = Threads.Atomic{Int}(0)
    report = ReentrantLock()
    stop = Threads.Atomic{Bool}(false)                     # set by the first task that fails
    failure = Ref{Any}(nothing)
    # Run the loop of one task. Its first error is written to the log and stops all tasks: the
    # others finish the momentum they are working on, and closing `built` releases the assembly
    # tasks that wait in `put!` (their InvalidStateException is then not an error of its own).
    guarded(work, name) = try
        work()
    catch err
        lock(report) do
            if !stop[]
                stop[] = true
                failure[] = CapturedException(err, catch_backtrace())
                @printf(log, "ERROR in %s task, stopping: %s\n", name, sprint(showerror, err)); flush(log)
            end
        end
        close(built)
    end
    build = map(1:builders) do _
        Threads.@spawn guarded("an assembly") do
            for i in todo
                stop[] && break
                time() - start < budget || continue
                window = Window(stack, path.q[i], W, L)
                present = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
                H = hamiltonian(model, window)
                drop > 0 && droptol!(H, drop)                  # entries with |h| ≤ drop (eV) are removed; ‖ΔH‖₂ ≤ largest dropped row sum
                H̃ = SlicedELL(UnrelaxedTTG.scaled(H, s, B_H), precision; reach = present)   # rows reachable from the zero labels
                put!(built, (i, H̃, present, length(window)))
            end
        end
    end
    consume = map(1:consumers) do _
        Threads.@spawn guarded("a GPU") do
            for (i, H̃, present, labels) in built
                stop[] && break
                t = time()
                H̃_gpu = gpu_matrix(H̃)
                μ = chebyshev_moments(H̃_gpu, present, P)
                CUDA.unsafe_free!(H̃_gpu)
                write_moments(file(i), μ)
                Threads.atomic_add!(done, 1)
                lock(report) do
                    @printf(log, "q %4d: %6d labels, %6d rows reached, %d zero labels, GPU %5.2f s, %7.0f s elapsed\n", i, labels,
                        size(H̃, 1), length(present) ÷ 2, time() - t, time() - start); flush(log)
                end
            end
        end
    end
    cpu = map(1:cpu_workers) do _                          # idle CPU threads take momenta from the same list
        Threads.@spawn guarded("a CPU") do
            for i in todo
                stop[] && break
                time() - start < budget || continue
                t = time()
                window = Window(stack, path.q[i], W, L)
                present = [o for t in 1:3 for o in orbitals(zero_label(window, t)) if zero_label(window, t) > 0]
                H = hamiltonian(model, window)
                drop > 0 && droptol!(H, drop)
                H̃ = SparseMatrixCSC{precision,Int}(UnrelaxedTTG.scaled(H, s, B_H))
                μ = chebyshev_moments(H̃, present, P)
                write_moments(file(i), μ)
                Threads.atomic_add!(done, 1)
                lock(report) do
                    @printf(log, "q %4d: %6d labels, %d zero labels, CPU %5.1f s, %7.0f s elapsed\n", i, length(window),
                        length(present) ÷ 2, time() - t, time() - start); flush(log)
                end
            end
        end
    end
    try
        foreach(wait, build); close(built); foreach(wait, consume); foreach(wait, cpu)
    catch err                                              # e.g. an interrupt (Ctrl-C) of this call: the tasks
        stop[] = true                                      # are not cancelled by it, so stop them and wait until
        close(built)                                       # they have finished their momentum, so that no task
        @printf(log, "interrupted (%s); waiting for the tasks to finish their momentum\n", typeof(err)); flush(log)
        for task in [build; consume; cpu]                  # still writes into the folder after this call returns
            try wait(task) catch end
        end
        rethrow()
    end
    if failure[] !== nothing                               # the moments written so far are kept; the same call resumes
        @printf(log, "stopped by an error after %d momenta in %.0f s; %d remain\n", done[], time() - start, length(pending) - done[]); flush(log)
        throw(failure[])
    end
    @printf(log, "finished %d momenta in %.0f s; %d remain\n", done[], time() - start, length(pending) - done[]); flush(log)
    return done[]
end

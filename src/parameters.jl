# Angle triples of the numerical experiments (manuscript, Section 5) and the bookkeeping of the
# output folders of the production drivers.

"""
    ANGLE_SETS

The angle triples used by the example drivers, with their default window radius W = (W₁ + W*)/2
and default scattering radius L (rule (5.5)); `ANGLE_SETS[set]` is `(θ, W, L)` with θ in degrees
and W, L in Å⁻¹:

| set | θ (degrees)        | W (Å⁻¹) | L (Å⁻¹) |
|-----|--------------------|---------|---------|
| 1   | (−1.4, 0, 2.8)     | 0.4742  | 19.40   |
| 2   | (−√3/2, 0, √2/2)   | 0.3771  | 30.56   |
| 3   | (−1.39, 0, 2.4)    | 0.4596  | 18.95   |
| 4   | (−1.39, 0, 2.8)    | 0.4744  | 19.555  |

Sets 1 and 2 are the triples of the manuscript with its values of W and L. Sets 3 and 4 follow
(5.3)–(5.5) evaluated for a = 1.42√3 Å (set 4: W₁ = 0.3102); the same evaluation gives 0.4748,
19.43 for set 1 and 0.3777, 30.60 for set 2, which differ from the manuscript's values in the
fourth digit.
"""
const ANGLE_SETS = (
    (θ = (-1.4, 0.0, 2.8), W = 0.4742, L = 19.40),
    (θ = (-√3 / 2, 0.0, √2 / 2), W = 0.3771, L = 30.56),
    (θ = (-1.39, 0.0, 2.4), W = 0.4596, L = 18.95),
    (θ = (-1.39, 0.0, 2.8), W = 0.4744, L = 19.555),
)

"""
    triple_parameters(set) -> (θ, W, L)

θ (degrees), default W and default L (Å⁻¹) of the angle triple number `set` of
[`ANGLE_SETS`](@ref). Throws an `ArgumentError` for any other `set`.
"""
function triple_parameters(set::Integer)
    set in eachindex(ANGLE_SETS) ||
        throw(ArgumentError("unknown angle set $set; the sets are 1, …, $(length(ANGLE_SETS)) (see ANGLE_SETS)"))
    return ANGLE_SETS[set].θ, ANGLE_SETS[set].W, ANGLE_SETS[set].L
end

"""
    claim_directory(directory, meta) -> file

Prepare the output folder of a run that can be resumed: create `directory` and write the
description `meta` of the run (the text of `meta.txt`: one `key value…` line per parameter) to
`directory/meta.txt`.

A resumed run skips the momenta whose moments `mu_*.bin` exist, so these must come from the same
parameters. If the folder already holds such files, its `meta.txt` must equal `meta` line by
line; otherwise an `ArgumentError` names the keys that differ (or says that `meta.txt` is
missing) and nothing is written. Line endings are not compared. A folder without moments is
taken over, and its `meta.txt` is replaced.
"""
function claim_directory(directory::AbstractString, meta::AbstractString)
    mkpath(directory)
    file = joinpath(directory, "meta.txt")
    if any(name -> startswith(name, "mu_") && endswith(name, ".bin"), readdir(directory))
        isfile(file) || throw(ArgumentError("$directory holds moments but no meta.txt, so their parameters are " *
            "unknown; use another folder"))
        lines(text) = split(replace(text, "\r\n" => "\n"), '\n'; keepempty = false)
        key(line) = first(split(line, ' '; limit = 2))
        old, new = lines(read(file, String)), lines(meta)
        if old != new
            entries(list) = Dict(key(line) => line for line in list)
            before, after = entries(old), entries(new)
            differing = sort!([k for k in union(keys(before), keys(after)) if get(before, k, nothing) != get(after, k, nothing)])
            what = isempty(differing) ? "the same lines in another order" : join(differing, ", ")
            throw(ArgumentError("$directory holds moments of a run with other parameters (differing in meta.txt: " *
                what * "); use another folder, or delete its mu_*.bin files to start again"))
        end
    end
    write(file, meta)
    return file
end

"""
    moments_complete(file, P) -> Bool

Whether `file` holds a finished set of P moments: it exists and has 8P bytes (P `Float64`). The
drivers compute a momentum again if its file is missing, empty or cut short (for example after a
power failure between the write and the rename of [`write_moments`](@ref)).
"""
moments_complete(file::AbstractString, P::Integer) = isfile(file) && filesize(file) == 8P

"""
    write_moments(file, μ) -> file

Writes the moments `μ` (raw `Float64`) to a temporary file of the calling task and renames it to
`file`, so that `file` is either absent or complete, and two tasks never write the same
temporary file. The temporary name does not end in `.bin`, so [`claim_directory`](@ref) does not
count it as moments.
"""
function write_moments(file::AbstractString, μ::AbstractVector{Float64})
    temporary = string(file, ".tmp", getpid(), "_", string(objectid(current_task()); base = 16))
    write(temporary, μ)
    mv(temporary, file; force = true)
    return file
end

"""
    run_jobs(f, jobs; budget = Inf, start = time()) -> number of jobs done

Calls `f(job)` for every element of `jobs` on the Julia threads (dynamic scheduling, as
`Threads.@threads :dynamic`). No job is started once `budget` seconds have passed since `start`, or
once a job has thrown: the jobs already running finish, then the first error is thrown again as a
`CapturedException` (the job is named in the message printed to `stderr`). Used by the CPU drivers
of `examples/`, so that one failing momentum stops the run instead of every other momentum being
computed first.
"""
function run_jobs(f, jobs; budget::Real = Inf, start::Real = time())
    done = Threads.Atomic{Int}(0)
    stop = Threads.Atomic{Bool}(false)
    failure = Ref{Any}(nothing)
    failure_lock = ReentrantLock()
    Threads.@threads :dynamic for job in collect(jobs)
        (stop[] || time() - start ≥ budget) && continue
        try
            f(job)
            Threads.atomic_add!(done, 1)
        catch err
            stop[] = true
            lock(failure_lock) do
                failure[] === nothing && (failure[] = (job, err, catch_backtrace()))
            end
        end
    end
    if failure[] !== nothing
        job, err, backtrace = failure[]
        println(stderr, "ERROR in job ", job, ", stopping after ", done[], " finished jobs")
        throw(CapturedException(err, backtrace))
    end
    return done[]
end

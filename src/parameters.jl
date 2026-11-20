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

# ---------------------------------------------------------------------------------------------
# Output folders: an exclusive lock, meta.txt, and moment files that survive an interruption.

"""Whether the process `pid` on this computer is running (a process we may not open counts as running)."""
function process_running(pid::Integer)
    pid > 0 || return false
    if Sys.iswindows()
        handle = ccall((:OpenProcess, "kernel32"), stdcall, Ptr{Cvoid}, (UInt32, Int32, UInt32), 0x1000, 0, pid)
        handle == C_NULL && return Libc.GetLastError() == 5          # ERROR_ACCESS_DENIED: it exists
        code = Ref{UInt32}(0)
        ok = ccall((:GetExitCodeProcess, "kernel32"), stdcall, Int32, (Ptr{Cvoid}, Ref{UInt32}), handle, code)
        ccall((:CloseHandle, "kernel32"), stdcall, Int32, (Ptr{Cvoid},), handle)
        return ok == 0 || code[] == 259                               # STILL_ACTIVE
    else
        return ccall(:kill, Cint, (Cint, Cint), pid, 0) == 0 || Libc.errno() == Libc.EPERM
    end
end

const HELD_LOCKS = Set{String}()              # lock folders taken by this process
const HELD_LOCKS_LOCK = ReentrantLock()
const RELEASE_AT_EXIT = Ref(false)

lock_folder(directory) = joinpath(abspath(directory), ".lock")

"""
    lock_directory(directory) -> lock folder

Takes the exclusive lock of the output folder `directory` (created if needed): the folder
`directory/.lock`, made with `mkdir`, which fails if it exists, so that of two runs started at
the same time in one folder only one proceeds. The file `.lock/owner` records the process id, the
computer and the time. A lock left behind by a process of this computer that no longer runs (for
example after a power failure) is taken over; any other existing lock throws an `ArgumentError`
that names it. Release it with [`unlock_directory`](@ref); locks still held when Julia exits
are removed then.
"""
function lock_directory(directory::AbstractString)
    mkpath(directory)
    folder = lock_folder(directory)
    for attempt in 1:2
        try
            mkdir(folder)
        catch err
            err isa Base.IOError || rethrow()
            isdir(folder) || rethrow()
            owner = joinpath(folder, "owner")
            text = isfile(owner) ? read(owner, String) : ""
            fields = split(text)
            pid = length(fields) ≥ 2 ? tryparse(Int, fields[2]) : nothing
            host = length(fields) ≥ 4 ? fields[4] : ""
            stale = pid !== nothing && host == gethostname() && pid != getpid() && !process_running(pid)
            if stale && attempt == 1
                rm(folder; recursive = true, force = true)
                continue
            end
            who = isempty(text) ? "an unknown run (no owner file)" :
                pid == getpid() ? "this Julia session" : "the run " * strip(text)
            throw(ArgumentError("$directory is in use by $who: another run is writing there. If no run uses " *
                "this folder any more, delete $folder"))
        end
        write(joinpath(folder, "owner"), "pid $(getpid()) host $(gethostname()) time $(round(Int, time()))\n")
        lock(HELD_LOCKS_LOCK) do
            push!(HELD_LOCKS, folder)
            if !RELEASE_AT_EXIT[]
                atexit(() -> foreach(f -> rm(f; recursive = true, force = true), collect(HELD_LOCKS)))
                RELEASE_AT_EXIT[] = true
            end
        end
        return folder
    end
end

"""
    unlock_directory(directory)

Releases the lock of `directory` taken by [`lock_directory`](@ref) in this process; does nothing
if this process does not hold it.
"""
function unlock_directory(directory::AbstractString)
    folder = lock_folder(directory)
    held = lock(HELD_LOCKS_LOCK) do
        folder in HELD_LOCKS ? (delete!(HELD_LOCKS, folder); true) : false
    end
    held && rm(folder; recursive = true, force = true)
    return nothing
end

"""
    claim_directory(directory, meta; lock = true) -> file

Prepare the output folder of a run that can be resumed: create `directory`, take its lock
([`lock_directory`](@ref), unless `lock = false`), and write the description `meta` of the run
(the text of `meta.txt`: one `key value…` line per parameter) to `directory/meta.txt`. The caller
releases the lock with [`unlock_directory`](@ref) when the run ends (the scripts of `examples/`
leave it to the exit of Julia).

A resumed run skips the momenta whose moments `mu_*.bin` exist, so these must come from the same
parameters. If the folder already holds such files, its `meta.txt` must equal `meta` line by
line; otherwise an `ArgumentError` names the keys that differ (or says that `meta.txt` is
missing), the lock is released and nothing is written. Line endings are not compared. A folder
without moments is taken over. `meta.txt` is written only when its text changes, through a
temporary file that is then renamed, so an interruption never leaves it empty.
"""
function claim_directory(directory::AbstractString, meta::AbstractString; lock::Bool = true)
    lock && lock_directory(directory)
    try
        mkpath(directory)
        file = joinpath(directory, "meta.txt")
        lines(text) = split(replace(text, "\r\n" => "\n"), '\n'; keepempty = false)
        old = isfile(file) ? read(file, String) : nothing
        if any(name -> startswith(name, "mu_") && endswith(name, ".bin"), readdir(directory))
            old === nothing && throw(ArgumentError("$directory holds moments but no meta.txt, so their parameters " *
                "are unknown; use another folder"))
            key(line) = first(split(line, ' '; limit = 2))
            before_lines, after_lines = lines(old), lines(meta)
            if before_lines != after_lines
                entries(list) = Dict(key(line) => line for line in list)
                before, after = entries(before_lines), entries(after_lines)
                differing = sort!([k for k in union(keys(before), keys(after)) if get(before, k, nothing) != get(after, k, nothing)])
                what = isempty(differing) ? "the same lines in another order" : join(differing, ", ")
                throw(ArgumentError("$directory holds moments of a run with other parameters (differing in meta.txt: " *
                    what * "); use another folder, or delete its mu_*.bin files to start again"))
            end
        end
        old == meta || replace_file(file, Vector{UInt8}(meta))
        return file
    catch
        lock && unlock_directory(directory)
        rethrow()
    end
end

"""Flush the data of the open file `f` to the disk (`fsync`, or `FlushFileBuffers` on Windows)."""
function sync_file(f::Base.Filesystem.File)
    ok = Sys.iswindows() ?
        ccall((:FlushFileBuffers, "kernel32"), stdcall, Int32, (Ptr{Cvoid},), f.handle) != 0 :
        ccall(:fsync, Cint, (Cint,), f.handle) == 0
    ok || throw(Base.IOError("could not flush a file to the disk", 0))
    return nothing
end

"""
Write `data` to `file` through a temporary file of the calling task: the data are written,
flushed to the disk, and the temporary file is renamed to `file`. A failing rename (on Windows a
new file can be held for a moment by a virus scanner or the indexer) is tried again five times
within about 3 s; on failure the temporary file is removed and the error is thrown.
"""
function replace_file(file::AbstractString, data)
    temporary = string(file, ".tmp", getpid(), "_", string(objectid(current_task()); base = 16))
    try
        f = Base.Filesystem.open(temporary, Base.Filesystem.JL_O_WRONLY | Base.Filesystem.JL_O_CREAT |
                                 Base.Filesystem.JL_O_TRUNC, 0o644)
        try
            write(f, data)
            sync_file(f)
        finally
            close(f)
        end
        for attempt in 0:5
            try
                mv(temporary, file; force = true)
                break
            catch err
                (err isa Base.IOError && attempt < 5) || rethrow()
                sleep(0.1 * 2^attempt)
            end
        end
    catch
        rm(temporary; force = true)
        rethrow()
    end
    return file
end

"""
    moments_complete(file, P) -> Bool

Whether `file` holds a finished set of P moments: it has 8P bytes (P `Float64`) and they are not
all zero. The drivers compute a momentum again if its file is missing, cut short, or all zeros.
[`write_moments`](@ref) never leaves a partial file; the zero test covers a power failure just
after its rename, after which a file system can show the new file with its full size but zero
content. A momentum without zero labels, whose moments are zero, is therefore always recomputed;
that costs only its window.
"""
function moments_complete(file::AbstractString, P::Integer)
    isfile(file) && filesize(file) == 8P || return false
    return any(!iszero, read!(file, Vector{Float64}(undef, P)))
end

"""
    write_moments(file, μ) -> file

Writes the moments `μ` (raw `Float64`) to a temporary file of the calling task, flushes it to
the disk and renames it to `file`, so that `file` is either absent or complete, and two tasks
never write the same temporary file. The temporary name does not end in `.bin`, so
[`claim_directory`](@ref) does not count it as moments.
"""
write_moments(file::AbstractString, μ::AbstractVector{Float64}) = replace_file(file, μ)

"""
    run_jobs(f, jobs; budget = Inf, start = time()) -> number of jobs done

Calls `f(job)` for every element of `jobs`, in their order, on `Threads.threadpoolsize()` tasks
that each take the next job as soon as they are free (so jobs sorted largest first are spread
over the threads). No job is started once `budget` seconds have passed since `start`, or once a
job has thrown: the jobs already running finish, then the first error is thrown again as a
`CapturedException` (the job is named in the message printed to `stderr`). An interrupt
(Ctrl-C) also lets the running jobs finish before it is thrown again. Used by the CPU drivers of
`examples/`.
"""
function run_jobs(f, jobs; budget::Real = Inf, start::Real = time())
    list = collect(jobs)
    next = Threads.Atomic{Int}(1)
    done = Threads.Atomic{Int}(0)
    stop = Threads.Atomic{Bool}(false)
    failure = Ref{Any}(nothing)
    failure_lock = ReentrantLock()
    function work()
        while !stop[] && time() - start < budget
            k = Threads.atomic_add!(next, 1)
            k > length(list) && return
            try
                f(list[k])
                Threads.atomic_add!(done, 1)
            catch err
                stop[] = true
                lock(failure_lock) do
                    failure[] === nothing && (failure[] = (list[k], err, catch_backtrace()))
                end
            end
        end
    end
    workers = [Threads.@spawn work() for _ in 1:min(Threads.threadpoolsize(), length(list))]
    try
        foreach(wait, workers)
    catch
        stop[] = true                                   # interrupted: start no new job, let the running ones finish
        for task in workers
            while !istaskdone(task)
                try wait(task) catch end
            end
        end
        rethrow()
    end
    if failure[] !== nothing
        job, err, backtrace = failure[]
        println(stderr, "ERROR in job ", job, ", stopping after ", done[], " finished jobs")
        throw(CapturedException(err, backtrace))
    end
    return done[]
end

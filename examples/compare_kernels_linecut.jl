# Jackson kernel against the Gaussian kernel on the stored moments of a line cut.
#
#   julia --project=. examples/compare_kernels_linecut.jl <directory> [P] [output file]
#
# <directory> holds the `meta.txt` and the moments `mu_<index>.bin` of `production_linecut.jl` or
# `production_linecut_gpu.jl`, with P_run moments per momentum; P (default P_run) is the number of
# moments used. On 801 energies in [−0.2, 0.2] eV the momentum LDoS (4.36) is evaluated with
#
#   J₁  the Jackson kernel, P moments (ε_P = π/(sP));
#   G₁  the Gaussian kernel, P moments, ε₁ = π/(sP) (the same nominal width);
#   G₂  the Gaussian kernel, P moments, ε₂ = 4/(sP) (P sε = 4);
#   J₂  the Jackson kernel with P₂ = round(π/(sε₂)) moments (the same nominal width as G₂);
#
# and, if P_run ≥ 2P, the references R₁ and R₂: the Gaussian kernel at ε₁ and ε₂ with all P_run
# moments, whose distance to the Gaussian LDoS of Section 2.5 is at most `truncation_bound`
# (printed). The differences are measured per momentum by (3.4), max_E |F − F_ref| / max_E |F_ref|,
# and summarised by the worst, median and best value over the momenta.
#
# Output: a text file with one `key value…` line per number (default `compare_kernels.txt` in the
# directory of the call).

using UnrelaxedTTG, Printf, Statistics

const E = range(-0.2, 0.2; length = 801)

"""s, the moments (P_run × momenta) and the arc lengths of a finished line-cut folder (throws if a momentum is not finished)."""
function read_linecut(directory)
    meta = Dict(split(line, ' '; limit = 2)[1] => split(line, ' '; limit = 2)[2] for line in eachline(joinpath(directory, "meta.txt")))
    P, s = parse(Int, meta["P"]), parse(Float64, meta["s"])
    arclength = parse.(Float64, split(meta["arclength"]))
    μ = zeros(P, length(arclength))
    buffer = Vector{Float64}(undef, P)
    unfinished = [i for i in eachindex(arclength) if !moments_complete(joinpath(directory, @sprintf("mu_%04d.bin", i)), P)]
    isempty(unfinished) || throw(ArgumentError("$directory: $(length(unfinished)) of $(length(arclength)) momenta are not " *
        "finished (the first is $(first(unfinished))); finish the run first"))
    for i in eachindex(arclength)
        read!(joinpath(directory, @sprintf("mu_%04d.bin", i)), buffer)
        μ[:, i] = buffer
    end
    return s, μ, arclength
end

"""Momentum LDoS (4.36), momenta × energies, from the first P moments with the kernel K."""
function ldos(μ, s, P, K)
    F = zeros(size(μ, 2), length(E))
    Threads.@threads for i in axes(μ, 2)
        F[i, :] = kpm_density(view(μ, 1:P, i), s, E; kernel = K) ./ 6
    end
    return F
end

"""Difference (3.4) per momentum: max_E |F − F_ref| / max_E |F_ref|."""
relative(F, F_ref) = [maximum(abs, view(F, i, :) .- view(F_ref, i, :)) / maximum(abs, view(F_ref, i, :)) for i in axes(F, 1)]

line(io, key, r::AbstractVector) = (@printf(io, "%s worst %.6e median %.6e best %.6e\n", key, maximum(r), median(r), minimum(r));
                                     @printf("%-10s worst %.3e  median %.3e  best %.3e\n", key, maximum(r), median(r), minimum(r)))
line(io, key, values...) = (println(io, key, " ", join(values, " ")); println(rpad(key, 10), " ", join(values, " ")))

directory = ARGS[1]
s, μ, arclength = read_linecut(directory)
P_run = size(μ, 1)
P = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : P_run
1 ≤ P ≤ P_run || throw(ArgumentError("P = $P must lie in 1, …, $P_run (the number of stored moments)"))
output = length(ARGS) ≥ 3 ? ARGS[3] : "compare_kernels.txt"
ε₁, ε₂ = π / (s * P), 4 / (s * P)
P₂ = round(Int, π / (s * ε₂))

open(output, "w") do io
    line(io, "directory", directory)
    line(io, "s_per_eV", s); line(io, "P", P); line(io, "P_run", P_run); line(io, "P2", P₂)
    line(io, "eps1_meV", 1e3ε₁); line(io, "eps2_meV", 1e3ε₂); line(io, "momenta", length(arclength))
    seconds = Dict{String,Float64}()
    ldos(μ, s, min(P, 16), Jackson()); ldos(μ, s, min(P, 16), Gaussian(ε₁))         # compile
    seconds["J1"] = @elapsed J₁ = ldos(μ, s, P, Jackson())
    seconds["G1"] = @elapsed G₁ = ldos(μ, s, P, Gaussian(ε₁))
    seconds["G2"] = @elapsed G₂ = ldos(μ, s, P, Gaussian(ε₂))
    seconds["J2"] = @elapsed J₂ = ldos(μ, s, P₂, Jackson())
    for key in ("J1", "G1", "G2", "J2")
        line(io, "seconds_" * key, seconds[key])
    end
    line(io, "threads", Threads.nthreads())
    line(io, "G1_vs_J1", relative(G₁, J₁))
    line(io, "G2_vs_J2", relative(G₂, J₂))
    line(io, "G2_vs_G1", relative(G₂, G₁))
    line(io, "max_J1", maximum(J₁)); line(io, "max_G1", maximum(G₁))
    for (key, F) in (("J1", J₁), ("G1", G₁), ("G2", G₂))
        i = argmin(F)
        line(io, "min_" * key, F[i], "momentum", i[1], "E_eV", E[i[2]], "fraction_negative", count(<(0), F) / length(F))
    end
    line(io, "truncation_bound_G1_per_eV", maximum(truncation_bound(Gaussian(ε₁), s, E, P)))
    line(io, "truncation_bound_G2_per_eV", maximum(truncation_bound(Gaussian(ε₂), s, E, P)))
    if P_run ≥ 2P
        R₁ = ldos(μ, s, P_run, Gaussian(ε₁))
        R₂ = ldos(μ, s, P_run, Gaussian(ε₂))
        line(io, "reference_bound_R1_per_eV", maximum(truncation_bound(Gaussian(ε₁), s, E, P_run)))
        line(io, "reference_bound_R2_per_eV", maximum(truncation_bound(Gaussian(ε₂), s, E, P_run)))
        line(io, "max_R1", maximum(R₁))
        line(io, "J1_vs_R1", relative(J₁, R₁))
        line(io, "G1_vs_R1", relative(G₁, R₁))
        line(io, "J2_vs_R2", relative(J₂, R₂))
        line(io, "G2_vs_R2", relative(G₂, R₂))
        line(io, "ratio_J1_over_G1", relative(J₁, R₁) ./ relative(G₁, R₁))
        line(io, "absolute_J1_vs_R1", maximum(abs, J₁ - R₁)); line(io, "absolute_G1_vs_R1", maximum(abs, G₁ - R₁))
        line(io, "absolute_J2_vs_R2", maximum(abs, J₂ - R₂)); line(io, "absolute_G2_vs_R2", maximum(abs, G₂ - R₂))
    end
end
println("wrote ", output)

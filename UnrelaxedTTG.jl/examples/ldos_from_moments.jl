# Momentum LDoS (4.36) from the moments written by `production_linecut.jl`.
#
#   julia --project=. examples/ldos_from_moments.jl <directory> <output file> [P′] [Emax] [number of energies]
#
# Output: header lines, then one line per finished momentum: index, arc length, LDoS at every energy (1/eV).

using UnrelaxedTTG, Printf

directory, output = ARGS[1], ARGS[2]
meta = Dict(split(line, ' '; limit = 2)[1] => split(line, ' '; limit = 2)[2] for line in eachline(joinpath(directory, "meta.txt")))
P = parse(Int, meta["P"]); s = parse(Float64, meta["s"])
P′ = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : P
Emax = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 0.25
E = range(-Emax, Emax; length = length(ARGS) ≥ 5 ? parse(Int, ARGS[5]) : 1001)
arclength = parse.(Float64, split(meta["arclength"]))
open(output, "w") do io
    println(io, "# E (eV): ", join(E, " "))
    println(io, "# ticks: ", meta["ticks"], " P ", P′, " resolution_eV ", kpm_resolution(s, P′))
    for i in eachindex(arclength)
        name = joinpath(directory, @sprintf("mu_%04d.bin", i))
        isfile(name) || continue
        μ = Vector{Float64}(undef, P); read!(name, μ)
        println(io, i, " ", arclength[i], " ", join(kpm_density(view(μ, 1:P′), s, E) ./ 6, " "))
    end
end
println("wrote ", output)

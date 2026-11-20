# Momentum LDoS (4.36) from the moments written by `production_linecut.jl` or `production_linecut_gpu.jl`.
#
#   julia --project=. examples/ldos_from_moments.jl <directory> <output file> [P′] [Emax] [number of energies] [eta=<η>] [m=<order>] [gauss=<ε>]
#
# Without `eta=` or `gauss=` the Jackson kernel of the manuscript is used. With `eta=<η>` (eV) the same
# moments are evaluated with the delta-Chebyshev kernel of order m (default 6) of arXiv:2512.03149,
# with `gauss=<ε>` (eV) with the Gaussian δ_ε expanded in Chebyshev polynomials; the header then also gives the largest value over the energies of `truncation_bound` (1/eV, for the
# LDoS, i.e. for μ₀/6 = 1), the bound for the terms beyond P′ that are left out.
#
# Output: header lines, then one line per finished momentum: index, arc length, LDoS at every energy (1/eV).

using UnrelaxedTTG, Printf
include(joinpath(@__DIR__, "kernel_argument.jl"))

kernel = kernel_argument!(ARGS)
directory, output = ARGS[1], ARGS[2]
meta = Dict(split(line, ' '; limit = 2)[1] => split(line, ' '; limit = 2)[2] for line in eachline(joinpath(directory, "meta.txt")))
P = parse(Int, meta["P"]); s = parse(Float64, meta["s"])
P′ = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : P
Emax = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 0.25
E = range(-Emax, Emax; length = length(ARGS) ≥ 5 ? parse(Int, ARGS[5]) : 1001)
arclength = parse.(Float64, split(meta["arclength"]))
open(output, "w") do io
    println(io, "# E (eV): ", join(E, " "))
    println(io, "# ticks: ", meta["ticks"], " P ", P′, " resolution_eV ", kpm_resolution(s, P′, kernel),
        kernel isa DeltaChebyshev ? " kernel delta-Chebyshev m $(length(kernel.z)) truncation_bound_per_eV $(maximum(truncation_bound(kernel, s, E, P′)))" :
        kernel isa Gaussian ? " kernel Gaussian truncation_bound_per_eV $(maximum(truncation_bound(kernel, s, E, P′)))" : "")
    for i in eachindex(arclength)
        name = joinpath(directory, @sprintf("mu_%04d.bin", i))
        isfile(name) || continue
        μ = Vector{Float64}(undef, P); read!(name, μ)
        println(io, i, " ", arclength[i], " ", join(kpm_density(view(μ, 1:P′), s, E; kernel) ./ 6, " "))
    end
end
println("wrote ", output, " (", kernel_text(kernel, s, P′), ")")

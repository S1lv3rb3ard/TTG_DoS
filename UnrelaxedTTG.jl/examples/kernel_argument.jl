# Choice of the regularisation kernel on the command line of the example scripts.

using UnrelaxedTTG, Printf

"""
    kernel_argument!(arguments) -> kernel

Removes the arguments of the form `eta=<η>` and `m=<order>` from `arguments` and returns the kernel
they select: without `eta=`, the Jackson kernel `Jackson()` of the manuscript; with `eta=<η>` (η in
eV), the delta-Chebyshev kernel `DeltaChebyshev(η, m)` of arXiv:2512.03149, of order m = 6 (the
order of the examples of that reference) unless `m=<order>` is given. The other arguments keep
their order.
"""
function kernel_argument!(arguments::Vector{String})
    value(key) = begin
        i = findfirst(startswith(key * "="), arguments)
        i === nothing ? nothing : split(popat!(arguments, i), '='; limit = 2)[2]
    end
    η, m = value("eta"), value("m")
    η === nothing && m !== nothing && throw(ArgumentError("m=$m needs eta=<η in eV>"))
    return η === nothing ? Jackson() : DeltaChebyshev(parse(Float64, η), m === nothing ? 6 : parse(Int, m))
end

"""Text for logs and file headers: the kernel and the parameter that sets its resolution."""
kernel_text(::Jackson, s, P) = @sprintf("Jackson kernel, ε_P = %.3f meV", 1e3 * kpm_resolution(s, P))
kernel_text(K::DeltaChebyshev, s, P) = @sprintf("delta-Chebyshev kernel of order %d, η = %.3f meV", length(K.z), 1e3 * K.η)

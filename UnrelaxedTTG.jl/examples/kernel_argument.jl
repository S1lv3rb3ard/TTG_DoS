# Choice of the regularisation kernel and of the interlayer model on the command line of the
# example scripts.

using UnrelaxedTTG, Printf

"""
    kernel_argument!(arguments) -> kernel

Removes the arguments of the form `eta=<η>`, `m=<order>` and `gauss=<ε>` from `arguments` and
returns the kernel they select: with none of them, the Jackson kernel `Jackson()` of the manuscript;
with `eta=<η>` (η in eV), the delta-Chebyshev kernel `DeltaChebyshev(η, m)` of arXiv:2512.03149, of
order m = 6 (the order of the examples of that reference) unless `m=<order>` is given; with
`gauss=<ε>` (ε in eV), the Gaussian δ_ε of width ε expanded in Chebyshev polynomials, `Gaussian(ε)`
(choose ε ≥ 4/(sP); the log line gives P sε). The other arguments keep their order.
"""
function kernel_argument!(arguments::Vector{String})
    value(key) = begin
        i = findfirst(startswith(key * "="), arguments)
        i === nothing ? nothing : split(popat!(arguments, i), '='; limit = 2)[2]
    end
    η, m, ε = value("eta"), value("m"), value("gauss")
    η === nothing && m !== nothing && throw(ArgumentError("m=$m needs eta=<η in eV>"))
    η !== nothing && ε !== nothing && throw(ArgumentError("eta= and gauss= select different kernels; give one of them"))
    ε === nothing || return Gaussian(parse(Float64, ε))
    return η === nothing ? Jackson() : DeltaChebyshev(parse(Float64, η), m === nothing ? 6 : parse(Int, m))
end

"""
    interlayer_argument!(arguments) -> name

Removes an argument of the form `interlayer=<name>` from `arguments` and returns the name of the
interlayer model as a `Symbol` (`:fang_kaxiras`, `:analytic_refit` or `:analytic_refit_6_7`; see
`UnrelaxedTTG.INTERLAYER_MODELS`), or `default` (`DEFAULT_INTERLAYER`, `:analytic_refit_6_7`) if
there is none. An unknown name throws.
"""
function interlayer_argument!(arguments::Vector{String}; default::Symbol = DEFAULT_INTERLAYER)
    i = findfirst(startswith("interlayer="), arguments)
    name = i === nothing ? default : Symbol(split(popat!(arguments, i), '='; limit = 2)[2])
    interlayer_model(name)                                    # throws for an unknown name
    return name
end

"""
    interlayer_meta(io, name)

Writes the line `interlayer <name>` of a meta.txt, except for `:fang_kaxiras`, so that the meta.txt
of a Fang–Kaxiras run is the same as before version 0.6.0 and older runs resume. (A meta.txt
without the line is a Fang–Kaxiras run, whatever the current default.)
"""
interlayer_meta(io::IO, name::Symbol) = name === :fang_kaxiras || println(io, "interlayer ", name)

# A model object: for a model of `INTERLAYER_MODELS` the same line as for its name (none for
# `FANG_KAXIRAS`), otherwise the object with all its parameters.
function interlayer_meta(io::IO, p::InterlayerModel)
    name = findfirst(==(p), INTERLAYER_MODELS)
    name === nothing ? println(io, "interlayer ", p) : interlayer_meta(io, name)
end

"""Short name of an interlayer model for logs: the name, or the orders (J₃, J₆) of an `AnalyticRefit`."""
interlayer_label(name::Symbol) = string(name)
function interlayer_label(p::InterlayerModel)
    name = findfirst(==(p), INTERLAYER_MODELS)
    name !== nothing && return string(name)
    p isa AnalyticRefit ? "AnalyticRefit of orders ($(length(p.v₃) - 1), $(length(p.v₆) - 1))" : string(typeof(p))
end

"""Text for logs and file headers: the kernel and the parameter that sets its resolution."""
kernel_text(::Jackson, s, P) = @sprintf("Jackson kernel, ε_P = %.3f meV", 1e3 * kpm_resolution(s, P))
kernel_text(K::DeltaChebyshev, s, P) = @sprintf("delta-Chebyshev kernel of order %d, η = %.3f meV", length(K.z), 1e3 * K.η)
kernel_text(K::Gaussian, s, P) = @sprintf("Gaussian kernel, ε = %.3f meV, P sε = %.2f", 1e3 * K.ε, P * s * K.ε)

# GPU version of the Chebyshev recurrence of Section 4.5 (loaded together with CUDA.jl).
#
# The extension adds two things to UnrelaxedTTG:
#
#   gpu_matrix(H̃[, T])                 the scaled matrix H̃ = sĤ_q^{(τ,W,L)} as a CuSparseMatrixCSR
#   chebyshev_moments(H̃_gpu, c, P)     the moments μ_n = ∑_c [T_n(H̃)]_cc, n = 0, …, P − 1, on the GPU
#
# Everything else (windows, assembly of Ĥ_q, KPM reconstruction (4.33)) stays on the CPU. The
# driver that uses them is examples/production_linecut_gpu.jl.
module UnrelaxedTTGCUDAExt

using UnrelaxedTTG, CUDA, LinearAlgebra, SparseArrays
using CUDA.CUSPARSE: CuSparseMatrixCSR
import UnrelaxedTTG: chebyshev_moments, gpu_matrix

# Docstring: `gpu_matrix` in src/kpm.jl. The upload is the matrix itself, not its transpose
# (for the Hermitian H̃ the moments would be the same either way, since T_n(H̃ᵀ) = T_n(H̃)ᵀ).
gpu_matrix(H̃::SparseMatrixCSC) = CuSparseMatrixCSR(H̃)
gpu_matrix(H̃::SparseMatrixCSC, ::Type{T}) where {T<:Number} = CuSparseMatrixCSR(SparseMatrixCSC{T,Int32}(H̃))   # e.g. T = ComplexF32

"""
    chebyshev_moments(H̃::CuSparseMatrixCSR, orbitals, P) -> μ::Vector{Float64}

Moments μ[n+1] = ∑_c [T_n(H̃)]_{cc}, n = 0, …, P − 1, of the start orbitals c ∈ `orbitals`, as in
the CPU method: with t_m = T_m(H̃) e_c and t_{m+1} = 2H̃ t_m − t_{m−1},

    μ_{2m} = 2⟨t_m, t_m⟩ − μ₀,      μ_{2m+1} = 2⟨t_{m+1}, t_m⟩ − μ₁.

H̃ must be Hermitian with σ(H̃) ⊂ (−1, 1) (see `UnrelaxedTTG.scaled`). The start vectors are the
columns of one dense n × c matrix on the GPU, so each step is one sparse-times-dense product
(CUSPARSE) and two inner products (CUBLAS); P moments cost ⌊P/2⌋ products (one for P = 1). The arithmetic is that
of the element type of H̃ (see [`gpu_matrix`](@ref)); the returned moments are `Float64`. The GPU
memory used is that of H̃ plus two n × c dense matrices, which are freed before returning.
"""
function chebyshev_moments(H̃::CuSparseMatrixCSR{T}, orbitals::AbstractVector{<:Integer}, P::Integer) where {T}
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    n, c = size(H̃, 1), length(orbitals)
    μ = zeros(P)
    c == 0 && return μ
    start = zeros(T, n, c)                                # t₀ on the CPU
    for (column, orbital) in enumerate(orbitals)
        start[orbital, column] = 1
    end
    previous = CuArray(start)                             # t₀
    current = similar(previous)
    mul!(current, H̃, previous)                           # t₁
    μ[1] = c
    P ≥ 2 && (μ[2] = real(dot(vec(previous), vec(current))))
    m = 1
    while 2m + 1 ≤ P                                      # here previous = t_{m−1}, current = t_m
        μ[2m+1] = 2 * real(dot(vec(current), vec(current))) - μ[1]
        2m + 2 ≤ P || break
        mul!(previous, H̃, current, T(2), T(-1))          # t_{m+1} = 2H̃ t_m − t_{m−1}, in place of t_{m−1}
        previous, current = current, previous
        μ[2m+2] = 2 * real(dot(vec(current), vec(previous))) - μ[2]
        m += 1
    end
    CUDA.unsafe_free!(previous); CUDA.unsafe_free!(current)
    return μ
end

end

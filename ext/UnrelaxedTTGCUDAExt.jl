# GPU version of the Chebyshev recurrence of Section 4.5 (loaded together with CUDA.jl).
#
# The extension adds two things to UnrelaxedTTG:
#
#   gpu_matrix(S)                      the scaled matrix H̃ = sĤ_q^{(τ,W,L)} in the storage S::SlicedELL, on the GPU
#   gpu_matrix(H̃[, T])                 the same matrix as a CuSparseMatrixCSR (CUSPARSE; slower, kept as reference)
#   chebyshev_moments(H̃_gpu, c, P)     the moments μ_n = ∑_c [T_n(H̃)]_cc, n = 0, …, P − 1, on the GPU, for both
#
# Everything else (windows, assembly of Ĥ_q, the storage S, KPM reconstruction (4.33)) stays on the
# CPU. The driver that uses them is examples/production_linecut_gpu.jl.
#
# Measured on an RTX 3080, 16 November 2026 (set 2, W = 0.25, L = 45.84, drop = 1e-5, ComplexF32,
# six start orbitals, P = 4000): 1.40 s per momentum with CuSparseMatrixCSR, 0.48 s with SlicedELL.
module UnrelaxedTTGCUDAExt

using UnrelaxedTTG, CUDA, LinearAlgebra, SparseArrays, StaticArrays
using CUDA.CUSPARSE: CuSparseMatrixCSR
using UnrelaxedTTG: SLICE, stored_orbitals
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

# ---------------------------------------------------------------------------------------------
# Recurrence on the sliced ELLPACK storage

# Docstring: `gpu_matrix` in src/kpm.jl. The index vectors `rows`, `position` stay on the CPU.
gpu_matrix(S::SlicedELL{T,<:Array}) where {T} =
    SlicedELL(S.rows, S.position, CuArray(S.offset), CuArray(S.width), CuArray(S.val), CuArray(S.col))

"""Free the GPU memory of a `SlicedELL` made by `gpu_matrix` (it must not be used afterwards)."""
CUDA.unsafe_free!(S::SlicedELL{T,<:CuArray}) where {T} = foreach(CUDA.unsafe_free!, (S.offset, S.width, S.val, S.col))

"""Threads of a block of [`step!`](@ref): a power of two (tree sum) and a multiple of `SLICE`."""
const THREADS = 1024

"""
    step!(next, current, offset, width, val, col, sums, k, α, β)

GPU kernel of one step of the recurrence, one thread per stored row i. With x = `current`,
y = `next` (vectors whose element i holds the C start columns of row i, C ≤ 6),

    y_i ← α ∑_e val[p] x_{col[p]} − β y_i,      p = offset[σ] + 32(e − 1) + r + 1,   e = 1, …, width[σ]

(row i has lane r in slice σ, see `SlicedELL`): y ← α H̃ x − β y. In the same pass the thread forms,
in `Float64`,

    a_i = ∑_c |y_ic|²,      b_i = ∑_c Re(conj(y_ic) x_ic)

for the new y, the block adds them up (pairwise, in a fixed order), and its first thread stores
the two sums in `sums[1, block, k]`, `sums[2, block, k]`. Summed over the blocks they are
‖y‖² and Re⟨y, x⟩, the inner products of the doubling relations.

The 32 threads of a warp read 32 neighbouring entries of `val` and `col` at each e. Nothing is
transferred to the CPU.
"""
function step!(next, current, offset, width, val, col, sums, k, α, β)
    thread = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + thread
    partial_a = CuStaticSharedArray(Float64, THREADS)
    partial_b = CuStaticSharedArray(Float64, THREADS)
    a = 0.0
    b = 0.0
    @inbounds if i ≤ length(next)
        σ = (i - 1) ÷ SLICE + 1
        base = offset[σ] + (i - 1) % SLICE + 1
        Hx = zero(eltype(next))
        for e in 0:width[σ]-1
            p = base + SLICE * e
            Hx += val[p] * current[col[p]]
        end
        x = current[i]
        y = α * Hx - β * next[i]
        next[i] = y
        for c in 1:length(y)
            a += Float64(abs2(y[c]))
            b += Float64(real(conj(y[c]) * x[c]))
        end
    end
    @inbounds partial_a[thread] = a
    @inbounds partial_b[thread] = b
    sync_threads()
    half = blockDim().x ÷ 2
    while half ≥ 1                                             # pairwise sum over the threads of the block
        if thread ≤ half
            @inbounds partial_a[thread] += partial_a[thread + half]
            @inbounds partial_b[thread] += partial_b[thread + half]
        end
        sync_threads()
        half ÷= 2
    end
    if thread == 1
        @inbounds sums[1, blockIdx().x, k] = partial_a[1]
        @inbounds sums[2, blockIdx().x, k] = partial_b[1]
    end
    return nothing
end

"""
    chebyshev_moments(H̃::SlicedELL on the GPU, orbitals, P) -> μ::Vector{Float64}

Moments μ[n+1] = ∑_c [T_n(H̃)]_{cc}, n = 0, …, P − 1, of the start orbitals c ∈ `orbitals` (rows of
the matrix that the `SlicedELL` was built from; they must be among its stored rows). With
t_k = T_k(H̃) e_c, t₁ = H̃ t₀ and t_k = 2H̃ t_{k−1} − t_{k−2}, step k = 1, …, ⌊P/2⌋ gives

    a_k = ∑_c ‖t_k‖²,      b_k = ∑_c Re⟨t_k, t_{k−1}⟩

and, with μ₀ = number of start orbitals and μ₁ = b₁, the doubling relations

    μ_{2k} = 2a_k − μ₀,      μ_{2k−1} = 2b_k − μ₁  (k ≥ 2).

Each step is one launch of the kernel [`step!`](@ref), which computes the product, the update and
the two inner products in one pass over the matrix. The inner products of all steps stay on the GPU
and are read once at the end, so the loop does not wait for the device. The products are formed in
the element type T of H̃ and the inner products are accumulated in `Float64`.

The result is the same, bit for bit, for every run with the same matrix, orbitals and P (the
sums have a fixed order). GPU memory: H̃, two vectors of n × c numbers and 16 ⌈n/1024⌉ ⌊P/2⌋ bytes.
"""
function chebyshev_moments(H̃::SlicedELL{T,<:CuArray}, orbitals::AbstractVector{<:Integer}, P::Integer) where {T}
    P ≥ 1 || throw(ArgumentError("P must be a positive integer"))
    isempty(orbitals) && return zeros(P)
    return moments(H̃, Val(length(orbitals)), stored_orbitals(H̃, orbitals), P)
end

function moments(H̃::SlicedELL{T}, ::Val{C}, start::Vector{Int}, P::Integer) where {T,C}
    n = size(H̃, 1)
    t₀ = zeros(SVector{C,T}, n)
    for (c, i) in enumerate(start)
        t₀[i] = setindex(t₀[i], one(T), c)
    end
    previous = CuArray(t₀)                                    # t₀
    current = CUDA.zeros(SVector{C,T}, n)                     # becomes t₁
    K = P ÷ 2                                                 # number of products
    blocks = cld(n, THREADS)
    sums = CUDA.zeros(Float64, 2, blocks, max(K, 1))
    if K ≥ 1
        R = real(T)
        matrix = (H̃.offset, H̃.width, H̃.val, H̃.col)
        # The tree sum needs blocks of exactly THREADS threads, so the kernel is compiled for THREADS
        # threads per block: the compiler then keeps to the register limit this allows (64 per thread on
        # current GPUs). C = 6 start columns in ComplexF64 need exactly 64; more columns spill to local
        # memory instead of failing (measured: C = 7, 64 registers, 1024 threads).
        kernel = @cuda launch = false maxthreads = THREADS step!(current, previous, matrix..., sums, 1, R(1), R(0))
        CUDA.maxthreads(kernel) ≥ THREADS || throw(ArgumentError("the recurrence kernel for $C start orbitals of " *
            "type $T was compiled with $(CUDA.registers(kernel)) registers per thread and allows at most " *
            "$(CUDA.maxthreads(kernel)) threads per block on this GPU, but its sums need blocks of $THREADS threads; " *
            "pass fewer orbitals per call and sum the moments of several calls"))
        kernel(current, previous, matrix..., sums, 1, R(1), R(0); threads = THREADS, blocks)      # t₁ = H̃ t₀  (current was 0)
        for k in 2:K                                          # here previous = t_{k−2}, current = t_{k−1}
            kernel(previous, current, matrix..., sums, k, R(2), R(1); threads = THREADS, blocks)  # t_k = 2H̃ t_{k−1} − t_{k−2}, in place of t_{k−2}
            previous, current = current, previous
        end
    end
    total = dropdims(sum(Array(sums); dims = 2); dims = 2)    # total[1, k] = a_k, total[2, k] = b_k; summed on the CPU in a fixed order
    CUDA.unsafe_free!(previous); CUDA.unsafe_free!(current); CUDA.unsafe_free!(sums)
    μ = zeros(P)
    μ[1] = C
    for k in 1:K
        μ[2k] = k == 1 ? total[2, 1] : 2 * total[2, k] - μ[2]
        2k + 1 ≤ P && (μ[2k+1] = 2 * total[1, k] - μ[1])
    end
    return μ
end

end

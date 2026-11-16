# Storage of the scaled Hamiltonian H̃ = sĤ_q^{(τ,W,L)} for the Chebyshev recurrence on a GPU
# (manuscript, Section 4.5): the rows that the recurrence can reach, in sliced ELLPACK form.
#
# Two facts are used.
#
# 1. The recurrence started at the orbitals c stays in the set R of rows connected to them by
#    entries of H̃ (`reachable`), and on R it only sees the block H̃_RR. The other rows of the
#    window do not influence the moments μ_n = ∑_c [T_n(H̃)]_cc and are not stored.
# 2. A GPU reads memory fastest when the 32 threads of a warp read 32 neighbouring entries. In
#    compressed-row storage the threads of a warp work on 32 rows that lie one after the other,
#    so they read entries that are one row length apart. `SlicedELL` stores every 32 consecutive
#    rows (a slice) entry by entry: first entry of the 32 rows, second entry of the 32 rows, ….

"""Number of rows of a slice of [`SlicedELL`](@ref): the 32 threads of a warp of the GPU."""
const SLICE = 32

"""
    reachable(H, orbitals) -> R::BitVector

The rows that the Chebyshev recurrence started at the orbitals c ∈ `orbitals` can occupy: `R[i]`
is true for the rows i of the smallest set R with

    c ∈ R,      j ∈ R and H_ij stored  ⇒  i ∈ R

(the rows connected to the start orbitals in the graph of `H`). H maps span{e_j : j ∈ R} into
itself, so T_n(H) e_c is supported in R for every n and

    [T_n(H)]_cc = [T_n(H_RR)]_cc,      c ∈ `orbitals`:

the moments of [`chebyshev_moments`](@ref) are those of the block H_RR. No symmetry of `H` is used.

For the windows of the production runs R is a proper subset: measured for set 2, W = 0.25,
L = 45.84 (16 November 2026), R holds 65 % of the rows and 98 % of the entries of Ĥ_q^{(τ,W,L)}.
"""
function reachable(H::SparseMatrixCSC, orbitals)
    n = LinearAlgebra.checksquare(H)
    R = falses(n)
    stack = Int[]
    for c in orbitals
        1 ≤ c ≤ n || throw(ArgumentError("the start orbital $c is not a row of the matrix"))
        R[c] || (R[c] = true; push!(stack, c))
    end
    rows = rowvals(H)
    while !isempty(stack)
        j = pop!(stack)
        for p in nzrange(H, j)
            i = rows[p]
            R[i] || (R[i] = true; push!(stack, i))
        end
    end
    return R
end

"""
    SlicedELL(H̃[, T]; reach = nothing, sort_window = 1024)

The square sparse matrix `H̃` with entries of type `T` (default: those of `H̃`) in sliced ELLPACK
storage, for the recurrence of [`chebyshev_moments`](@ref) on a GPU (upload with
[`gpu_matrix`](@ref)).

Rows. With `reach = orbitals` only the rows of R = [`reachable`](@ref)`(H̃, orbitals)` are stored,
that is, the block H̃_RR, which gives the same moments for these start orbitals; without it, all
rows. The stored rows are taken in the order of `H̃`, and inside every `sort_window` consecutive
rows they are sorted by decreasing number of entries, so that the 32 rows of a slice have nearly
the same length. This is a symmetric permutation Π H̃_RR Πᵀ; `rows[i]` is the row of `H̃` stored as
row i, and `position[j]` the stored row of row j of `H̃` (0 if it is not stored).

Entries. Slice number σ = 1, 2, … holds the stored rows 32(σ − 1) + 1, …, 32σ, padded with zeros
to the length `width[σ]` of its longest row. Entry number e = 1, …, `width[σ]` of the row with
lane r = 0, …, 31 in the slice is at

    p = offset[σ] + 32(e − 1) + r + 1:      value `val[p]`, column `col[p]` (a stored row).

Padding entries have the value 0 and the column 1. `offset[end]` = `length(val)`.

`SparseMatrixCSC(S)` returns Π H̃_RR Πᵀ. `size(S)` is its size.
"""
struct SlicedELL{T,V<:AbstractVector{T},I<:AbstractVector{Int32}}
    rows::Vector{Int}
    position::Vector{Int}
    offset::I
    width::I
    val::V
    col::I
end

function SlicedELL(H̃::SparseMatrixCSC, ::Type{T} = eltype(H̃); reach = nothing, sort_window::Integer = 1024) where {T}
    N = LinearAlgebra.checksquare(H̃)
    sort_window ≥ 1 || throw(ArgumentError("sort_window must be positive"))
    R = reach === nothing ? trues(N) : reachable(H̃, reach)
    rowval, nzval = rowvals(H̃), nonzeros(H̃)
    # Number of entries of every row of H̃_RR. The columns j ∈ R have all their rows in R.
    entries = zeros(Int, N)
    for j in 1:N
        R[j] || continue
        for p in nzrange(H̃, j)
            entries[rowval[p]] += 1
        end
    end
    rows = findall(R)
    n = length(rows)
    for lo in 1:sort_window:n                                  # stable sort: rows of equal length keep their order
        sort!(view(rows, lo:min(lo + sort_window - 1, n)); by = i -> entries[i], rev = true, alg = MergeSort)
    end
    position = zeros(Int, N)
    position[rows] = 1:n
    slices = cld(n, SLICE)
    offset = zeros(Int32, slices + 1)
    width = zeros(Int32, slices)
    stored = 0
    for σ in 1:slices
        offset[σ] = stored
        width[σ] = maximum(i -> entries[rows[i]], SLICE * (σ - 1) + 1:min(SLICE * σ, n))
        stored += SLICE * Int(width[σ])
        stored ≤ typemax(Int32) || throw(ArgumentError("more than 2³¹ − 1 stored entries"))
    end
    offset[end] = stored
    val = zeros(T, stored)
    col = ones(Int32, stored)                                  # padding: value 0, column 1
    filled = zeros(Int32, n)                                   # entries written so far in each stored row
    for j in 1:N
        R[j] || continue
        column = position[j]
        for p in nzrange(H̃, j)
            i = position[rowval[p]]
            σ, r = divrem(i - 1, SLICE)
            e = (filled[i] += Int32(1))
            q = offset[σ + 1] + SLICE * (e - 1) + r + 1
            val[q] = nzval[p]
            col[q] = column
        end
    end
    return SlicedELL(rows, position, offset, width, val, col)
end

Base.size(S::SlicedELL) = (length(S.rows), length(S.rows))
Base.size(S::SlicedELL, d::Integer) = d ≤ 2 ? length(S.rows) : 1
Base.eltype(::SlicedELL{T}) where {T} = T

function SparseArrays.SparseMatrixCSC(S::SlicedELL{T,<:Array}) where {T}
    n = size(S, 1)
    I, J, V = Int[], Int[], T[]
    for i in 1:n
        σ, r = divrem(i - 1, SLICE)
        for e in 1:S.width[σ + 1]
            p = S.offset[σ + 1] + SLICE * (e - 1) + r + 1
            iszero(S.val[p]) || (push!(I, i); push!(J, S.col[p]); push!(V, S.val[p]))
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    stored_orbitals(S, orbitals) -> Vector{Int}

The stored rows `S.position[c]` of the orbitals c (rows of the original matrix). Throws an
`ArgumentError` if one of them is not stored (it is then not in the set R of the `reach` that
`S` was built with).
"""
function stored_orbitals(S::SlicedELL, orbitals::AbstractVector{<:Integer})
    all(c -> 1 ≤ c ≤ length(S.position) && S.position[c] > 0, orbitals) ||
        throw(ArgumentError("a start orbital is not among the stored rows; build the SlicedELL with `reach` containing it"))
    return S.position[orbitals]
end

# On the CPU the recurrence runs on the compressed-column form of the stored block (same moments).
chebyshev_moments(S::SlicedELL{T,<:Array}, orbitals::AbstractVector{<:Integer}, P::Integer) where {T} =
    chebyshev_moments(SparseMatrixCSC(S), stored_orbitals(S, orbitals), P)

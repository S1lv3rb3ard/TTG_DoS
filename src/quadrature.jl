# Quadrature rules and the momentum cutoff.

"""
    gauss_legendre(n) -> (x, w)

Nodes and weights of the n-point Gauss–Legendre rule on [−1, 1] (Golub–Welsch:
eigenvalues of the Jacobi matrix with off-diagonal β_m = 1/(2√(1 − (2m)⁻²))).
"""
function gauss_legendre(n::Integer)
    n ≥ 1 || throw(ArgumentError("the order must be a positive integer"))
    β = [1 / (2 * sqrt(1 - (2m)^-2.0)) for m in 1:n-1]
    x, V = eigen(SymTridiagonal(zeros(n), β))
    return x, 2 .* abs2.(V[1, :])
end

"""
    polar_quadrature(nᵣ, n_φ, R) -> (x, w)

Rule for ∫_{|x|<R} f(x) dx in polar coordinates: Gauss–Legendre in the radius
and the periodic trapezoid rule in the angle. Nodes x = r (cos φ, sin φ);
the weights include the Jacobian r.
"""
function polar_quadrature(nᵣ::Integer, n_φ::Integer, R::Real)
    R > 0 || throw(ArgumentError("the cutoff radius must be positive"))
    ξ, wξ = gauss_legendre(nᵣ)
    r = @. R * (ξ + 1) / 2
    wᵣ = @. R * wξ / 2 * r
    φ = [2π * (b - 1) / n_φ for b in 1:n_φ]
    x = [Vec2(r[a] * cos(φ[b]), r[a] * sin(φ[b])) for b in 1:n_φ for a in 1:nᵣ]
    w = [wᵣ[a] * 2π / n_φ for b in 1:n_φ for a in 1:nᵣ]
    return x, w
end

"""
    momentum_cutoff(r, τ, δ)

The smooth bump g_τ of eq. (4.4) at ‖x‖₂ = `r`: 1 for r ≤ τ − δ, 0 for r ≥ τ, and

    e^{−δ/(τ−r)} / (e^{−δ/(τ−r)} + e^{−δ/(r−(τ−δ))})

in between.
"""
@inline function momentum_cutoff(r::Real, τ::Real, δ::Real)
    r ≤ τ - δ && return one(float(r))
    r ≥ τ && return zero(float(r))
    left = exp(-δ / (τ - r))
    right = exp(-δ / (r - (τ - δ)))
    return left / (left + right)
end

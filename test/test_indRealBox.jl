using LinearAlgebra
using ProximalOperators
import ProximalCore
using Test

@testset "IndRealBox" begin

f = IndRealBox(-1.0, 2.0)

@test ProximalCore.is_set_indicator(f) == true
@test ProximalCore.is_convex(f) == true
@test ProximalCore.is_separable(f) == true

# inside the box and real: indicator is zero, prox is the identity
x = ComplexF64[-1.0, 0.0, 0.5, 2.0]
@test f(x) == 0.0
y, fy = prox(f, x)
@test y == x
@test fy == 0.0

# a nonzero imaginary part is outside the set even where the real part is in range
@test f(ComplexF64[0.5 + 1.0im]) == Inf
# and so is a real part out of range
@test f(ComplexF64[2.5 + 0.0im]) == Inf

# the projection discards the imaginary part, then clamps the real one
x = ComplexF64[-3.0+2.0im, 0.25-1.0im, 7.0+0.0im]
y, fy = prox(f, x)
@test y == ComplexF64[-1.0, 0.25, 2.0]
@test fy == 0.0
@test all(iszero, imag.(y))
y_naive, fy_naive = ProximalOperators.prox_naive(f, x, 1.0)
@test y ≈ y_naive
@test fy ≈ fy_naive

# elementwise bounds
lb = [-1.0, 0.0]
ub = [1.0, 0.0]
g = IndRealBox(lb, ub)
y, _ = prox(g, ComplexF64[2.0+1.0im, -3.0-1.0im])
@test y == ComplexF64[1.0, 0.0]

@test_throws ErrorException IndRealBox(1.0, -1.0)
@test_throws ErrorException IndRealBox([0.0, 0.0], [1.0, 1.0, 1.0])

end

@testset "IndRealNonnegative" begin

f = IndRealNonnegative()
@test f isa IndRealBox

x = ComplexF64[-2.0+1.0im, 0.0, 3.0-4.0im]
y, fy = prox(f, x)
@test y == ComplexF64[0.0, 0.0, 3.0]
@test fy == 0.0

@test f(ComplexF64[0.0, 1.0]) == 0.0
@test f(ComplexF64[-1.0]) == Inf
@test f(ComplexF64[1.0im]) == Inf

end

using LinearAlgebra
using ProximalOperators
import ProximalCore
using Test

@testset "IndBallL0" begin

f = IndBallL0(3)

predicates_test(f)

@test ProximalCore.is_set_indicator(f) == true
@test ProximalCore.is_convex(f) == false

x = [0.5, -3.0, 0.1, 2.0, -1.0, 0.05]

call_test(f, x)
prox_test(f, x, 0.7)

y, fy = prox(f, x)
@test fy == 0.0
@test count(!iszero, y) == 3
kept = findall(!iszero, y)
@test y[kept] == x[kept]
# nothing dropped may be larger than anything kept
@test maximum(abs, x[setdiff(eachindex(x), kept)]) <= minimum(abs, x[kept])

# `prox!` must also be correct when it writes into its own input, which callers are allowed to do
x_inplace = copy(x)
prox!(x_inplace, f, x_inplace, 1.0)
@test x_inplace == y

# the same, for a complex array
z = ComplexF64[0.1 + 0.1im, -3.0 + 1.0im, 0.2, 2.0 - 2.0im]
g = IndBallL0(2)
w, _ = prox(g, z)
z_inplace = copy(z)
prox!(z_inplace, g, z_inplace, 1.0)
@test z_inplace == w
@test count(!iszero, z_inplace) == 2

end

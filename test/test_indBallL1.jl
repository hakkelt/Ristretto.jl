using LinearAlgebra
using Random
using ProximalOperators
import ProximalCore
using Test

@testset "IndBallL1" begin

f = IndBallL1(1.0)

predicates_test(f)

@test ProximalCore.is_set_indicator(f) == true
@test ProximalCore.is_convex(f) == true

@testset "membership" begin
    @test f([0.4, -0.3, 0.2]) == 0.0      # strictly inside
    @test f([0.5, -0.5]) == 0.0           # exactly on the boundary
    @test f([0.5, -0.5, 1e-6]) == Inf     # clearly outside
    @test f(ComplexF64[0.3 + 0.4im, 0.5]) == 0.0
    @test f(ComplexF64[0.3 + 0.4im, 0.6]) == Inf

    g = IndBallL1(1.0f0)
    @test g(Float32[0.5, -0.5]) == 0.0f0
    @test g(Float32[0.5, -0.5, 1.0f-3]) == Inf32
end

@testset "the prox lands inside the ball" begin
    x = [0.5, -3.0, 0.1, 2.0, -1.0]
    prox_test(f, x, 1.0)
    y, fy = prox(f, x, 1.0)
    @test fy == 0.0
    @test f(y) == 0.0
    @test norm(y, 1) ≈ 1.0
end

@testset "a point produced through Precompose is still in the ball" begin
    # Regression: the membership tolerance used to be `f.r*eps(R)`, tight enough that the output of
    # this function's own prox could be reported as outside the ball. Composing with a linear mapping
    # leaves a rounding error of a few `n*eps` in the ℓ1 norm — orders of magnitude above `eps(R)` —
    # so `g(prox(g, x))` returned `Inf` for the great majority of random inputs.
    rng = MersenneTwister(0)
    for _ in 1:50
        Q = Matrix(qr(randn(rng, 10, 10)).Q)
        g = Precompose(f, Q, 1.0)
        x = randn(rng, 10)
        y, fy = prox(g, x, 1.0)
        @test fy == 0.0
        @test g(y) == 0.0
    end
end

end

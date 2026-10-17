using LinearAlgebra
using ProximalOperators
import ProximalCore
using Test

@testset "SeparableHuberLoss" begin

rho, mu = 1.5, 0.7
f = SeparableHuberLoss(rho, mu)

predicates_test(f)

@test ProximalCore.is_smooth(f) == true
@test ProximalCore.is_separable(f) == true
@test ProximalCore.is_convex(f) == true
@test ProximalCore.is_quadratic(f) == false
@test ProximalCore.is_set_indicator(f) == false

huber(t) = abs(t) <= rho ? mu / 2 * abs(t)^2 : rho * mu * (abs(t) - rho / 2)

# entries on both sides of the kink
x = [0.0, 0.3, -1.4, 1.5, -1.6, 4.0]

call_test(f, x)
prox_test(f, x, 1.3)
grad_fx, fx = gradient_test(f, x)

@test abs(fx - f(x)) <= 1e-12
@test abs(f(x) - sum(huber, x)) <= 1e-12

# the gradient is mu*t below the kink and rho*mu*sign(t) above it
expected_grad = [abs(t) <= rho ? mu * t : rho * mu * sign(t) for t in x]
@test norm(expected_grad - grad_fx, Inf) <= 1e-12

# the prox is the element-wise scalar Huber prox
for gamma in [0.1, 0.9, 3.0]
    y, fy = prox(f, x, gamma)
    expected = [abs(t) <= rho * (1 + mu * gamma) ? t / (1 + mu * gamma) : t - gamma * mu * rho * sign(t) for t in x]
    @test norm(y - expected, Inf) <= 1e-12
    @test abs(fy - f(y)) <= 1e-12
    prox_test(f, x, gamma)
end

# on a single-entry array it agrees with the non-separable HuberLoss
x1 = [0.4]
@test f(x1) ≈ HuberLoss(rho, mu)(x1)
@test prox(f, x1, 0.7)[1] ≈ prox(HuberLoss(rho, mu), x1, 0.7)[1]

# complex input: the loss depends on the entries only through their magnitude
z = ComplexF64[0.3 + 0.4im, -2.0 + 1.0im, 0.0]
call_test(f, z)
prox_test(f, z, 0.8)
@test abs(f(z) - sum(huber ∘ abs, z)) <= 1e-12

# limiting behaviour: rho -> 0 approaches rho*mu*‖x‖₁, rho -> Inf approaches mu/2*‖x‖²
x2 = randn(50)
@test abs(SeparableHuberLoss(1e-8, mu)(x2) / (1e-8 * mu) - norm(x2, 1)) <= 1e-5
@test abs(SeparableHuberLoss(1e8, mu)(x2) - mu / 2 * norm(x2)^2) <= 1e-8

# a Float32 problem stays in Float32
x32 = randn(Float32, 20)
f32 = SeparableHuberLoss(1.0f0, 0.5f0)
@test typeof(f32(x32)) == Float32
y32, fy32 = prox(f32, x32, 0.5f0)
@test eltype(y32) == Float32
@test typeof(fy32) == Float32

end

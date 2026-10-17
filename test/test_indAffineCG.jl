using ProximalOperators
import ProximalCore
using LinearAlgebra
using Test

# A deliberately matrix-free operator: it exposes only `*` and `'`, which is all IndAffineCG may use.
struct OpWrapper{M}
    A::M
end
Base.:*(L::OpWrapper, x) = L.A * x
Base.adjoint(L::OpWrapper) = OpWrapperAdjoint(L.A)
struct OpWrapperAdjoint{M}
    A::M
end
Base.:*(L::OpWrapperAdjoint, y) = L.A' * y

@testset "IndAffineCG" begin

    T = Float64

    @testset "projects onto {x : Ax = b}" begin
        A = T[1.0 2.0 3.0 0.5; 0.0 1.0 -1.0 2.0]
        b = T[1.0, -0.5]
        x = T[0.3, -1.2, 2.0, 0.7]

        f = IndAffineCG(OpWrapper(A), b; maxit = 200, tol = 1.0e-12)

        @test ProximalCore.is_affine_indicator(f) == true

        y, fy = prox(f, x, T(1))
        @test fy == T(0)
        @test A * y ≈ b atol = 1.0e-9
        # The projection is the least-norm correction, i.e. it agrees with the pseudo-inverse formula.
        @test y ≈ x + A' * ((A * A') \ (b - A * x)) atol = 1.0e-8
        # ... and a point already in the set is left alone.
        y2, = prox(f, y, T(1))
        @test y2 ≈ y atol = 1.0e-9
    end

    @testset "indicator value" begin
        A = T[1.0 0.0; 0.0 1.0]
        b = T[0.0, 0.0]
        f = IndAffineCG(OpWrapper(A), b; tol = 1.0e-8)
        @test f(T[0.0, 0.0]) == T(0)
        @test f(T[1.0, 0.0]) == T(Inf)
    end

    @testset "AAc_diag skips the CG iteration" begin
        # A with orthonormal rows: AA' = I, so the closed form and CG must agree exactly.
        A = T[1.0 0.0 0.0; 0.0 0.0 1.0]
        b = T[0.5, -1.0]
        x = T[0.3, -1.2, 2.0]

        cg = IndAffineCG(OpWrapper(A), b; maxit = 200, tol = 1.0e-12)
        # `maxit = 0` proves no CG iteration is run: with the diagonal shortcut the answer is still exact.
        diagonal = IndAffineCG(OpWrapper(A), b; maxit = 0, tol = 1.0e-12, AAc_diag = T(1))

        y_cg, = prox(cg, x, T(1))
        y_diag, = prox(diagonal, x, T(1))
        @test y_diag ≈ y_cg atol = 1.0e-9
        @test A * y_diag ≈ b atol = 1.0e-12

        # A vector diagonal is accepted too.
        A2 = T[2.0 0.0 0.0; 0.0 0.0 3.0]
        d = T[4.0, 9.0]   # diag(A2 * A2')
        f2 = IndAffineCG(OpWrapper(A2), b; maxit = 0, AAc_diag = d)
        y2, = prox(f2, x, T(1))
        @test A2 * y2 ≈ b atol = 1.0e-12

        # A zero diagonal entry means that row is dropped rather than divided by.
        A3 = T[0.0 0.0 0.0; 0.0 0.0 3.0]
        f3 = IndAffineCG(OpWrapper(A3), b; maxit = 0, AAc_diag = T[0.0, 9.0])
        y3, = prox(f3, x, T(1))
        @test all(isfinite, y3)
        @test (A3 * y3)[2] ≈ b[2] atol = 1.0e-12
    end

    @testset "prox! may write into its own input" begin
        A = T[1.0 2.0 3.0 0.5; 0.0 1.0 -1.0 2.0]
        b = T[1.0, -0.5]
        x = T[0.3, -1.2, 2.0, 0.7]
        f = IndAffineCG(OpWrapper(A), b; maxit = 200, tol = 1.0e-12)
        expected, = prox(f, x, T(1))
        z = copy(x)
        prox!(z, f, z, T(1))
        @test z ≈ expected
    end

end

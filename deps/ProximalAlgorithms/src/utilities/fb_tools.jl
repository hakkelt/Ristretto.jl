

function f_model(f_x, grad_f_x, res, L)
    return f_x - real(dot(grad_f_x, res)) + (L / 2) * norm(res)^2
end

function lower_bound_smoothness_constant(f, A, x, grad_f_Ax)
    R = real(eltype(x))
    xeps = x .+ 1
    f_Axeps, grad_f_Axeps = value_and_gradient(f, A * xeps)
    return norm(A' * (grad_f_Axeps - grad_f_Ax)) / R(sqrt(length(x)))
end

function lower_bound_smoothness_constant(f, A, x)
    R = real(eltype(x))
    Ax = A * x
    f_Ax, grad_f_Ax = value_and_gradient(f, Ax)
    return lower_bound_smoothness_constant(f, A, x, grad_f_Ax)
end

_mul!(y, L, x) = mul!(y, L, x)
_mul!(y, ::Nothing, x) = return

# The squared norms the safeguard needs, `‖x - x'‖²`, `‖∇ - ∇'‖²`, `‖∇‖²`, `‖∇'‖²`, `‖x‖²` and
# `‖x'‖²`, as one reduction over the four arrays: without materializing the differences, without
# indexing scalars so that it works on device arrays, and with a single synchronization there (an
# `ArrayPartition` adds the sums of its parts, in the RecursiveArrayTools extension). The
# sums are at least double precision, as BLAS `nrm2` accumulates: a reduction that is not pairwise
# rounds a single-precision sum enough to move the quotient across `1/gamma`.
_safeguard_sums(::Type{R}, x, x_prev, grad, grad_prev) where {R} =
    map(R, _safeguard_sums_in(promote_type(R, Float64), x, x_prev, grad, grad_prev))

function _safeguard_sums_in(::Type{S}, x, x_prev, grad, grad_prev) where {S}
    term(u, u_prev, v, v_prev) = (
        S(abs2(u - u_prev)), S(abs2(v - v_prev)), S(abs2(v)), S(abs2(v_prev)), S(abs2(u)), S(abs2(u_prev)),
    )
    terms = Broadcast.instantiate(Broadcast.broadcasted(term, x, x_prev, grad, grad_prev))
    return mapreduce(identity, (a, b) -> map(+, a, b), terms; init = ntuple(_ -> zero(S), Val(6)))
end

# Host arrays: the generic reduction cannot vectorize a tuple accumulator, so a loop keeps six.
function _safeguard_sums_in(::Type{S}, x::Array, x_prev::Array, grad::Array, grad_prev::Array) where {S}
    s1 = s2 = s3 = s4 = s5 = s6 = zero(S)
    @inbounds @simd for i in eachindex(x, x_prev, grad, grad_prev)
        u, u_prev, v, v_prev = x[i], x_prev[i], grad[i], grad_prev[i]
        s1 += S(abs2(u - u_prev))
        s2 += S(abs2(v - v_prev))
        s3 += S(abs2(v))
        s4 += S(abs2(v_prev))
        s5 += S(abs2(u))
        s6 += S(abs2(u_prev))
    end
    return (s1, s2, s3, s4, s5, s6)
end

# The secant safeguard of a fixed stepsize. `L`-smoothness of `f` means `‖∇f(x) - ∇f(x')‖ ≤ L ‖x - x'‖`
# for every pair of points, so the gradients an accelerated method evaluates anyway give a lower bound
# on `L` at every iteration. A fixed `gamma = 1/Lf` from an `Lf` that came out low is exactly what this
# bound can expose, at the cost of one pass over four arrays and two copies per iteration, and no
# extra evaluation of `f`. The quotient is computed from rounded gradients, so the rounding that can
# accumulate in them is subtracted first: `noise` bounds it by a small multiple of the unit roundoff
# at the gradients' and the iterates' scales. Returns the stepsize to continue with: `gamma` itself
# unless the observed quotient exceeds `1/gamma`, and then `1/(1.01 L)` for the observed `L`, since
# the quotient is only a lower bound on the true constant.
function lipschitz_safeguard(gamma::R, x, x_prev, grad, grad_prev) where {R}
    s = _safeguard_sums(R, x, x_prev, grad, grad_prev)
    dx = sqrt(s[1])
    dx > 0 || return gamma
    noise = 64 * eps(R) * (sqrt(s[3]) + sqrt(s[4]) + (sqrt(s[5]) + sqrt(s[6])) / gamma)
    L = (sqrt(s[2]) - noise) / dx
    L * gamma > 1 || return gamma
    return R(1 / (R(1.01) * L))
end

function backtrack_stepsize!(
    gamma::R,
    f,
    A,
    g,
    x,
    f_Ax::R,
    At_grad_f_Ax,
    y,
    z,
    g_z::R,
    res,
    Az,
    grad_f_Az = nothing;
    alpha = R(1),
    minimum_gamma = R(1e-7),
    reduce_gamma = R(0.5),
) where {R}
    f_Az_upp = f_model(f_Ax, At_grad_f_Ax, res, alpha / gamma)
    _mul!(Az, A, z)
    f_Az, grad_f_Az_tmp = value_and_gradient(f, Az)
    tol = 10 * eps(R) * (1 + abs(f_Az))
    while f_Az > f_Az_upp + tol && gamma >= minimum_gamma
        gamma *= reduce_gamma
        y .= x .- gamma .* At_grad_f_Ax
        g_z = prox!(z, g, y, gamma)
        res .= x .- z
        f_Az_upp = f_model(f_Ax, At_grad_f_Ax, res, alpha / gamma)
        _mul!(Az, A, z)
        f_Az, grad_f_Az_tmp = value_and_gradient(f, Az)
        tol = 10 * eps(R) * (1 + abs(f_Az))
    end
    if grad_f_Az !== nothing
        grad_f_Az .= grad_f_Az_tmp
    end
    if gamma < minimum_gamma
        @warn "stepsize `gamma` became too small ($(gamma))"
    end
    return gamma, g_z, f_Az, f_Az_upp
end

function backtrack_stepsize!(
    gamma::R,
    f,
    A,
    g,
    x;
    alpha = R(1),
    minimum_gamma = R(1e-7),
    reduce_gamma = R(0.5),
) where {R}
    Ax = A * x
    f_Ax, grad_f_Ax = value_and_gradient(f, Ax)
    At_grad_f_Ax = A' * grad_f_Ax
    y = x - gamma .* At_grad_f_Ax
    z, g_z = prox(g, y, gamma)
    return backtrack_stepsize!(
        gamma,
        f,
        A,
        g,
        x,
        f_Ax,
        At_grad_f_Ax,
        y,
        z,
        g_z,
        x - z,
        Ax,
        grad_f_Ax;
        alpha = alpha,
        minimum_gamma = minimum_gamma,
        reduce_gamma = reduce_gamma,
    )
end

"""
    Component(name::Symbol, regs::Regularization...)

One additive component of an image decomposition (e.g. the low-rank part of an
`L+S` reconstruction). `name` is mandatory and must be unique among the components
passed to `reconstruct`. At least one regularization is required.

# Example
```julia
julia> using Ristretto
julia> Component(:lowrank, LowRank(0.05; time_dim = :time), L1TemporalFourier(0.01; time_dim = :time))
Component(:lowrank, LowRank(0.05), L1TemporalFourier(0.01))
```
"""
struct Component{R <: Tuple{Vararg{Regularization}}}
    name::Symbol
    regularizations::R
    function Component(name::Symbol, regs::Regularization...)
        @argcheck !isempty(regs) "Component `$name` needs at least one regularization."
        return new{typeof(regs)}(name, regs)
    end
end

function Base.show(io::IO, c::Component)
    print(io, "Component(:", c.name)
    for reg in c.regularizations
        print(io, ", ", reg)
    end
    return print(io, ")")
end

"""
    check_components(components::Tuple{Vararg{Component}})

Validate a tuple of `Component`s: at least two components, all unique names.
Throws `ArgumentError` otherwise.
"""
function check_components(components::Tuple{Vararg{Component}})
    @argcheck length(components) >= 2 "Image decomposition needs at least two components; use the plain regularization API for a single component."
    names = map(c -> c.name, components)
    @argcheck length(unique(names)) == length(names) "Component names must be unique, got $names."
    collisions = filter(in(_RECON_IMAGE_RESERVED_NAMES), names)
    @argcheck isempty(collisions) "Component name(s) $collisions collide with ReconImage's own field(s) $_RECON_IMAGE_RESERVED_NAMES; rename the component(s)."
    return nothing
end

function get_affected_dims(c::Component, acq_info::Union{Nothing, AcquisitionInfo}, image_dims)
    dims = Any[]
    for reg in c.regularizations
        append!(dims, get_affected_dims(reg, acq_info, image_dims))
    end
    return unique(dims)
end

function scale_regularization(c::Component, factor::Real)
    return Component(c.name, map(reg -> scale_regularization(reg, factor), c.regularizations)...)
end

function bind_dimensions(c::Component, image_dims)
    return Component(c.name, map(reg -> bind_dimensions(reg, image_dims), c.regularizations)...)
end

function materialize(c::Component, x::Variable; threaded::Bool)
    terms, _ = materialize_with_auxiliaries(c, x; threaded)
    return terms
end

function materialize_with_auxiliaries(c::Component, x::Variable; threaded::Bool)
    # The constructor guarantees at least one regularization, so the reduction needs no seed.
    term_list, auxiliaries = materialize_all(c.regularizations, x; threaded)
    return reduce(+, term_list), auxiliaries
end

function rescale!(img::ReconImage, factor)
    unname(parent(img)) .*= factor
    for c in components(img)
        unname(c) .*= factor
    end
    return img
end

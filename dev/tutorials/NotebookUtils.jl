#=
NotebookUtils.jl

Shared preamble for the tutorials in docs/literate/ and the notebooks made from them. Every
tutorial's first code cell does:

    include("NotebookUtils.jl")
    using .NotebookUtils

This module exists so the 12 tutorials do not each re-fix the same two display glitches, and do
not each re-implement the same three or four tiny plotting helpers. docs/make.jl copies it next
to the generated notebooks.
=#
module NotebookUtils

using MIRTjim: jim, jim!
# `Plots` itself, not only the names used here: `@animate` expands to code that refers to
# `Plots.Animation`, so the module has to be resolvable inside this one.
import Plots
using Plots: Plot, @animate, gif
using LinearAlgebra: norm
using Pkg: Pkg
using InteractiveUtils: versioninfo

export nrmse, side_by_side, difference_image, grid_layout, print_versions, kaxes, animate_slices,
    animate_frames

# --------------------------------------------------------------------------------------------
# 1. jim orientation: MIRTjim's `yflip` default is `minimum(y) >= 0`, which is `true` for the
#    plain pixel-index axes every notebook uses (`y = 1:size(z, 2)`). That default renders the
#    Shepp-Logan / torso phantoms upside down relative to how they are conventionally shown
#    (the standard picture has the off-centre gray "ventricle" ellipse in the UPPER half and the
#    small high-contrast "tumour" markers in the LOWER half — verified visually against this
#    package's own `create_shepp_logan_phantom` output, see the A0 handoff report). Overriding
#    the *global* default here — instead of passing `yflip = false` at every call site — is the
#    single point of control the campaign plan asks for: it fixes every `jim` call in every
#    notebook, present and future, without touching the phantom arrays themselves.
jim(:yflip, false)

# Every image panel carries axis labels, the same way every plot does. `x`/`y` are the image-domain
# pixel axes, which is what most panels show; a panel showing something else (k-space, a sampling
# mask over `kx`/`ky`, a Casorati matrix) passes `xlabel`/`ylabel` explicitly and overrides these.
# Setting them as global defaults here is the same single point of control as `yflip` above, rather
# than repeating two keywords at every `jim` call in twelve notebooks.
jim(:xlabel, "x")
jim(:ylabel, "y")

# GR sizes a subplot's plot area first and draws the axis labels into whatever is left, so in a
# multi-panel figure the leftmost panel's `ylabel` and the bottom row's `xlabel` are drawn outside
# the canvas and simply do not appear. Reserving the margin globally is the same single point of
# control as the two defaults above: it fixes every figure in every notebook rather than adding two
# keywords to each `plot` and `jim` call that happens to be wide enough to clip.
#
# The right margin is the same defect seen from the other side: a colour bar's tick labels are
# drawn outside the subplot, so a panel whose values need long labels (`0.0000125`) loses the last
# characters off the edge of the canvas.
Plots.default(left_margin = 6Plots.mm, bottom_margin = 8Plots.mm, right_margin = 10Plots.mm)

"""
    kaxes

The `xlabel`/`ylabel` pair for a panel that shows k-space rather than the image domain — a raw
k-space array, a sampling mask, a point-spread function's source. Splat it into the `jim` call
(`jim(mask; title = "...", kaxes...)`) so every k-space panel in every notebook is labelled the
same way, rather than each one spelling out two keywords.

A view along other k-space axes (`kx`/`kz`, say) passes its own labels; a `NamedDimsArray` takes
its labels from its own dimension names and needs neither.
"""
const kaxes = (xlabel = "kx", ylabel = "ky")

# --------------------------------------------------------------------------------------------
# 2. Write plot titles and axis labels in plain ASCII (`title = "Ax"`, not `title = "𝒜x"`).
#    Script letters stay in markdown prose and as Julia variable names.
#
# --------------------------------------------------------------------------------------------
# 3. Shared helpers the notebooks otherwise redefine ad hoc (surveyed across all 12 notebooks
#    before writing these): a one- or two-line `nrmse(x̂) = norm(abs.(x̂) - abs.(x_true)) /
#    norm(abs.(x_true))` closure-over-x_true appears in notebooks 04/05/09 (`nrmse`), 07
#    (`nrmse_dyn`) and 08 (`aligned_nrmse`); side-by-side `jim` panels built by hand from
#    `p1 = jim(...); p2 = jim(...); jim(p1, p2; layout = (1, 2))` appear throughout without a
#    shared color scale; and a `title = "Error"` difference panel (`jim(abs.(x̂ - x); title =
#    "Error")`) appears repeatedly (notebook 01 and others). These three replace those.

"""
    nrmse(x̂, x) -> Real

Normalized RMSE between a reconstruction `x̂` and a reference `x`, magnitude-only (so it is
meaningful for complex images):

    nrmse(x̂, x) = ‖|x̂| - |x|‖₂ / ‖|x|‖₂

This is the two-argument, referentially-transparent form; a notebook that repeatedly compares
against the same ground truth may still define a local one-argument closure
`nrmse1(x̂) = nrmse(x̂, x_true)` for brevity.
"""
nrmse(x̂, x) = norm(abs.(x̂) .- abs.(x)) / norm(abs.(x))

"""
    grid_layout(n; maxcols = 3) -> (rows, cols)

Row/column counts for a figure of `n` panels, with at most `maxcols` panels per row. Notebook
figures cap a row at three images so that panels stay legible in the exported HTML at ordinary
screen widths; a `jim` call that renders several slices or frames *inside one panel* is not
affected by this and may show more.
"""
function grid_layout(n::Integer; maxcols::Integer = 3)
    n >= 1 || throw(ArgumentError("grid_layout: need at least one panel, got $n"))
    rows = cld(n, min(n, maxcols))
    return (rows, cld(n, rows))   # balance the rows: four panels are 2x2, not 3+1
end

"""
    side_by_side(images...; titles = ("", "", ...), clim = nothing, size = ..., maxcols = 3, kwargs...)

Display several images side by side (via `jim`/`MIRTjim`) on a SHARED color scale, so the panels
are visually comparable. By default `clim` is the joint `(min, max)` of `abs.(image)` across all
`images`; pass `clim` explicitly to override, or `clim = :each` to let every panel scale itself.
Use `:each` when the panels are not in the same units — a real-valued image next to a log-magnitude
spectrum, say, where one shared scale flattens the panel with the smaller range to a flat field.
`titles` pairs with `images` positionally. Remaining `kwargs` are forwarded to every panel's `jim`
call (not to the combining `jim`).

At most `maxcols` panels go in one row; with more images the figure wraps onto further rows
(see [`grid_layout`](@ref)), and the default `size` grows with the row count accordingly.
"""
function side_by_side(
        images...;
        titles = ntuple(_ -> "", length(images)),
        clim = nothing,
        maxcols::Integer = 3,
        size = nothing,
        kwargs...,
    )
    length(titles) == length(images) ||
        throw(ArgumentError("side_by_side: got $(length(images)) images but $(length(titles)) titles"))
    panels = if clim === :each
        [jim(img; title = t, kwargs...) for (img, t) in zip(images, titles)]
    else
        mags = map(img -> abs.(img), images)
        shared_clim = clim === nothing ? (minimum(minimum, mags), maximum(maximum, mags)) : clim
        [jim(img; title = t, clim = shared_clim, kwargs...) for (img, t) in zip(images, titles)]
    end
    rows, cols = grid_layout(length(images); maxcols = maxcols)
    figsize = size === nothing ? (350 * cols, 350 * rows) : size
    return jim(panels...; layout = (rows, cols), size = figsize)
end

"""
    animate_slices(volume; dim = ndims(volume), fps = 6, title = i -> "", clim = nothing, kwargs...)

An animated GIF stepping along `dim` of `volume` — the right display for a stack of slices or a
dynamic series, where a montage of static panels forces the reader to compare frames by eye.

Every frame shares one colour scale (the magnitude range over the whole volume unless `clim` says
otherwise), because a per-frame scale makes the animation flicker and hides exactly the intensity
changes a dynamic series is about. `title` is a function of the frame index. Remaining `kwargs` go
to `jim`.

The returned value renders inline in the notebook and is embedded in the exported HTML, so nothing
is written next to the page.
"""
function animate_slices(
        volume; dim::Integer = ndims(volume), fps::Real = 6,
        title = _ -> "", clim = nothing, kwargs...,
    )
    frames = Array(volume)                       # drops NamedDims wrappers; `selectdim` wants a plain array
    mags = abs.(frames)
    shared = clim === nothing ? (minimum(mags), maximum(mags)) : clim
    anim = @animate for i in axes(frames, dim)
        jim(selectdim(frames, dim, i); title = title(i), clim = shared, kwargs...)
    end
    return gif(anim; fps, show_msg = false)
end

"""
    animate_frames(render, nframes; fps = 6)

An animated GIF whose `i`-th frame is whatever `render(i)` returns — the general form of
[`animate_slices`](@ref), for an animation whose every frame is a *figure* rather than a single
panel (several volumes side by side, an image next to the curve it comes from).

`render` is responsible for its own colour scales; pass an explicit `clim` to each panel, since
anything left to autoscale is recomputed per frame and the animation flickers.

```julia
animate_frames(nt) do i
    jim(jim(A[:, :, i]; clim = ca), jim(B[:, :, i]; clim = cb); layout = (1, 2))
end
```
"""
function animate_frames(render, nframes::Integer; fps::Real = 6)
    anim = @animate for i in 1:nframes
        render(i)
    end
    return gif(anim; fps, show_msg = false)
end

"""
    difference_image(x̂, x; title = "Error", scale = 1, kwargs...)

Display the magnitude difference image `abs.(x̂ .- x)` with ITS OWN color scale (errors are
typically much smaller in magnitude than the images being compared, so sharing `clim` with them
would make the error panel look uniformly black). `scale` multiplies the difference before
display, e.g. to make a small error visible; the title records the scale factor when `scale != 1`.
`kwargs` are forwarded to `jim`.
"""
function difference_image(x̂, x; title = "Error", scale = 1, kwargs...)
    d = scale .* abs.(x̂ .- x)
    displayed_title = scale == 1 ? title : "$title (×$scale)"
    return jim(d; title = displayed_title, kwargs...)
end

# --------------------------------------------------------------------------------------------
# 4. Version cell: the last cell of every notebook, so an exported HTML page records exactly
#    what produced it. Read from the active manifest via `Pkg.dependencies()` rather than
#    hardcoded, so it can't go stale the way a copy-pasted version string would.

const DEV_PATHED_DEPS = (
    "AbstractOperators", "FFTWOperators", "NFFTOperators", "WaveletOperators", "DSPOperators",
    "ContourletOperators", "StructuredOptimization", "ProximalOperators", "ProximalAlgorithms",
    "OperatorCore", "NestedThreading",
)

"""
    print_versions()

Print `versioninfo()`, `Ristretto`'s own version, and the version of every
dev-pathed fork under `deps/` (`AbstractOperators`, `NestedThreading`, and friends — see
`DEV_PATHED_DEPS`), each tagged `(dev)` when it is resolved to a local path rather than a
registry release. Intended as the final cell of every notebook.
"""
function print_versions()
    versioninfo()
    println()
    deps = Pkg.dependencies()
    by_name = Dict(info.name => info for info in values(deps))
    for name in ("Ristretto", DEV_PATHED_DEPS...)
        info = get(by_name, name, nothing)
        if info === nothing
            println(rpad(name, 24), "not loaded")
            continue
        end
        tag = info.is_tracking_path ? " (dev)" : ""
        println(rpad(name, 26), something(info.version, "unversioned"), tag)
    end
    return nothing
end

end # module

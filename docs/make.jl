using Documenter
using Literate
using Base64: base64decode
using Pkg: Pkg
using SHA: SHA
include(joinpath(@__DIR__, "benchmark_tables.jl"))  # BenchmarkTables, for related_packages.md
using Ristretto
using Ristretto.ProximalAlgorithms
using Ristretto.ProximalOperators
using Ristretto.ContourletOperators

# The extension surface is `public` but not exported (see NAMING.md §6), so bring the names
# documented here into scope for the `@docs` and `@ref` blocks that reference them unqualified.
using Ristretto: Regularization, ReconstructionMethod, IterativeMethod,
    DirectMethod, Scaling, CoilCombination, DataFidelity, Verbosity,
    ReconstructionExecutor, Subsampling, VariableDensityDistribution, PartialFourierFilter,
    DensityCompensation, CoilCompression, SensitivityEstimation, GradientDelay,
    get_operator, materialize, materialize_with_auxiliaries, materialize_all, get_affected_dims,
    scale_regularization, bind_dimensions, calculate, check_applicable,
    get_encoding_operator, get_fourier_operator, get_sensitivity_map_operator, get_subsampling_operator,
    build_encoding_operator, signal_model_operator, NamedDimsOp, DEFAULT_ALGORITHMS

# Internals whose docstrings are rendered on the low-level pages, or that other docstrings link to
# with `@ref`. Cross-references resolve in the page's module, so these have to be in scope too.
using Ristretto: with_serial_blas, serial_blas_threshold_bytes,
    set_serial_blas_threshold_bytes!,
    model_encoding_operator, StackedNSCTOp, BlockNuclearNorm, DenoiserProx

# Tutorials: each Literate script in docs/literate/ becomes an executed page under
# src/tutorials/ and a notebook (not executed) to download next to it. RISTRETTO_DOCS_TUTORIALS
# selects some of them by number ("01,09"), for a quick local build; "none" skips them.
#
# Literate executes the scripts, not Documenter, so an executed page can be kept: next to each
# page a stamp records the hash of everything its output depends on, and a page whose stamp
# still matches is not executed again. RISTRETTO_DOCS_CACHE=off executes every page anyway.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")
const UTILS = joinpath(LITERATE_DIR, "NotebookUtils.jl")
const ROOT = dirname(@__DIR__)

function selected_tutorials()
    all = sort(filter(f -> occursin(r"^\d\d_.*\.jl$", f), readdir(LITERATE_DIR)))
    sel = get(ENV, "RISTRETTO_DOCS_TUTORIALS", "all")
    sel == "all" && return all
    sel == "none" && return String[]
    keep = split(sel, ',')
    return filter(f -> any(k -> startswith(f, lpad(k, 2, '0')), keep), all)
end

# The page loads the helper module from its source; the notebook expects it next to itself.
# Code marked `#nb` (a timing measurement, whose page shows a snapshot instead) runs in the
# notebook only; the page shows it as a plain code block that is not executed.
function page_source(content, name)
    content = replace(content, "include(\"NotebookUtils.jl\")" => "include($(repr(UTILS))) #hide")
    content = replace(
        content, r"(?m)(^#nb (?!# ).*\n)+" => s -> "# ```julia\n" * replace(s, r"(?m)^#nb " => "# ") * "# ```\n"
    )
    download = "#md # *Download this tutorial as a [Jupyter notebook]($(name).ipynb) (with " *
        "[`NotebookUtils.jl`](NotebookUtils.jl) next to it).*\n#md #\n"
    first_line, rest = split(content, '\n'; limit = 2)
    return first_line * "\n#md #\n" * download * rest
end

# The scripts write LaTeX the way notebooks read it, `$x$` and `$$x$$`. Documenter's Markdown
# takes a `$` as interpolation, and an inline formula broken over lines as two of them, so the
# page gets its own math syntax: ``x`` inline, a math block for display. Code blocks are kept.
function documenter_math(md)
    parts = split(md, r"(?m)(?=^```)")
    in_code = false
    for (i, part) in enumerate(parts)
        if startswith(part, "```")
            # A fence opens a block unless one is open, in which case it closes it and the text
            # after it is prose again.
            if in_code
                fence, prose = split(part, '\n'; limit = 2)
                parts[i] = fence * "\n" * _math_prose(prose)
            end
            in_code = !in_code
        else
            parts[i] = _math_prose(part)
        end
    end
    return join(parts)
end

function _math_prose(text)
    text = replace(text, r"\$\$(.+?)\$\$"s => s -> "```math\n" * strip(s[3:(end - 2)]) * "\n```")
    return replace(text, r"\$([^$]+?)\$"s => s -> "``" * replace(s[2:(end - 1)], r"\s*\n\s*" => " ") * "``")
end

# What every executed page depends on besides its own script: the package and the vendored
# dependencies, the resolved environment, this file (it shapes the page), the helper module,
# the Julia version and the thread count (the timings print it). One hash per input, so a page
# that is executed again can say which inputs changed.
function common_inputs()
    files = String[joinpath(ROOT, f) for f in ("Project.toml", "Manifest.toml")]
    append!(files, (joinpath(@__DIR__, f) for f in ("Project.toml", "make.jl")))
    push!(files, UTILS)
    for dir in ("src", "ext", "deps"), (root, _, fs) in walkdir(joinpath(ROOT, dir)), f in sort(fs)
        push!(files, joinpath(root, f))
    end
    inputs = Pair{String, String}[relpath(f, ROOT) => source_hash(f) for f in files if isfile(f)]
    versions = sort!(["$(d.name) $(d.version) $(d.tree_hash)" for d in values(Pkg.dependencies())])
    push!(inputs, "packages" => bytes2hex(SHA.sha256(join(versions, '\n'))))
    push!(inputs, "julia" => "$(VERSION)_threads=$(Threads.nthreads())")
    return inputs
end

# Julia source is hashed by what it parses to, without comments, docstrings, line numbers or
# formatting, none of which reaches a page: editing a docstring does not execute the tutorials
# again. A file that does not parse is hashed as it is.
function source_hash(path)
    endswith(path, ".jl") || return bytes2hex(SHA.sha256(read(path)))
    ex = try
        Meta.parseall(read(path, String); filename = path)
    catch
        return bytes2hex(SHA.sha256(read(path)))
    end
    ctx = SHA.SHA256_CTX()
    hash_source!(ctx, ex)
    return bytes2hex(SHA.digest!(ctx))
end

hash_source!(ctx, ::LineNumberNode) = nothing
hash_source!(ctx, x::QuoteNode) = (SHA.update!(ctx, codeunits("quote ")); hash_source!(ctx, x.value))
hash_source!(ctx, x) = SHA.update!(ctx, codeunits(repr(x) * " "))
function hash_source!(ctx, ex::Expr)
    # `"doc" f(x) = ...` parses to `Core.@doc(line, "doc", f(x) = ...)`
    if Meta.isexpr(ex, :macrocall) && ex.args[1] == GlobalRef(Core, Symbol("@doc")) && length(ex.args) == 4
        return hash_source!(ctx, ex.args[4])
    end
    SHA.update!(ctx, codeunits("($(ex.head) "))
    foreach(a -> hash_source!(ctx, a), ex.args)
    SHA.update!(ctx, codeunits(") "))
    return
end

stamp_lines(inputs) = ["$h $name" for (name, h) in inputs]

# The inputs a stamp lists that are new, gone or different now.
function changed_inputs(stamp, inputs)
    old = Dict(
        let (h, name) = split(l, ' '; limit = 2)
            name => h
        end for l in readlines(stamp)[2:end] if !isempty(l)
    )
    new = Dict(name => h for (name, h) in inputs)
    return sort!([name for name in union(keys(old), keys(new)) if get(old, name, "") != get(new, name, "")])
end

# The pages of tutorials that no longer exist would still be rendered.
function remove_orphaned_pages()
    names = Set(splitext(f)[1] for f in readdir(LITERATE_DIR))
    for f in readdir(TUTORIAL_DIR)
        m = match(r"^(\d\d_[^.]*?)(-(img)?\d+)?\.(md|png|gif|jpeg|svg|stamp|ipynb)$", f)
        isnothing(m) || m[1] in names || rm(joinpath(TUTORIAL_DIR, f))
    end
    return
end

# Literate inlines a result that can show itself as HTML, which plots and animations can, as a
# base64 data URI; written to files instead, they keep the page under Documenter's size limit.
function extract_images(md, name)
    n = Ref(0)
    pattern = r"```@raw html\n<img src=\"data:image/(png|gif|jpeg|svg\+xml);base64,([A-Za-z0-9+/=]+)\" />\n```"
    return replace(
        md, pattern => function (block)
            m = match(pattern, block)
            file = "$(name)-img$(n[] += 1).$(m[1] == "svg+xml" ? "svg" : m[1])"
            write(joinpath(TUTORIAL_DIR, file), base64decode(m[2]))
            return "![]($file)"
        end
    )
end

function build_tutorials()
    mkpath(TUTORIAL_DIR)
    remove_orphaned_pages()
    cp(UTILS, joinpath(TUTORIAL_DIR, "NotebookUtils.jl"); force = true)
    use_cache = get(ENV, "RISTRETTO_DOCS_CACHE", "on") != "off"
    common = common_inputs()
    pages = Pair{String, String}[]
    for file in selected_tutorials()
        name = splitext(file)[1]
        path = joinpath(LITERATE_DIR, file)
        # The script itself is hashed as written: its comments are the page's prose.
        inputs = [common; relpath(path, ROOT) => bytes2hex(SHA.sha256(read(path)))]
        key = bytes2hex(SHA.sha256(join(stamp_lines(inputs), '\n')))
        stamp = joinpath(TUTORIAL_DIR, name * ".stamp")
        page = joinpath(TUTORIAL_DIR, name * ".md")
        if use_cache && isfile(page) && isfile(stamp) && readline(stamp) == key
            @info "Tutorial $name: unchanged, page kept"
        else
            if isfile(stamp)
                changed = changed_inputs(stamp, inputs)
                shown = join(first(changed, 10), ", ") * (length(changed) > 10 ? ", …" : "")
                @info "Tutorial $name: inputs changed ($(length(changed))): $shown"
            end
            rm(stamp; force = true)
            for f in readdir(TUTORIAL_DIR)
                startswith(f, name * "-") && rm(joinpath(TUTORIAL_DIR, f))
            end
            t = @elapsed Literate.markdown(
                path, TUTORIAL_DIR; documenter = true, execute = true, credit = false,
                preprocess = c -> page_source(c, name), postprocess = md -> extract_images(documenter_math(md), name),
            )
            write(stamp, join([key; stamp_lines(inputs)], '\n') * '\n')
            @info "Tutorial $name executed in $(round(Int, t)) s"
        end
        Literate.notebook(path, TUTORIAL_DIR; execute = false, credit = false)
        title = replace(match(r"^# # (.*)$"m, read(path, String))[1], r"^\d+ — " => "")
        push!(pages, title => joinpath("tutorials", name * ".md"))
    end
    return pages
end

const TUTORIAL_PAGES = build_tutorials()

makedocs(;
    modules = [Ristretto, ProximalAlgorithms, ProximalOperators, ContourletOperators],
    authors = "Tamás Hakkel <hakkelt@gmail.com>",
    sitename = "Ristretto.jl",
    format = Documenter.HTML(
        assets = [asset("assets/favicon.svg", class = :ico, islocal = true)]
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => TUTORIAL_PAGES,
        "High-level Interface" => [
            "AcquisitionInfo" => "high-level/acquisition_info.md",
            "Preprocessing" => "high-level/preprocessing.md",
            "Simulation Tools" => "high-level/simulation.md",
            "Reconstruction Methods" => "high-level/methods.md",
            "Reconstruction" => "high-level/reconstruction.md",
            "Export" => "high-level/export.md",
            "Regularization" => "high-level/regularization.md",
            "Optimization Algorithms" => "high-level/algorithms.md",
            "Task Splitting" => "high-level/task_splitting.md",
            "Image Decomposition" => "high-level/image_decomposition.md",
            "Noise & Analysis" => "high-level/analysis.md",
            "Performance & Threading" => "high-level/performance.md",
            "GPU Reconstruction" => "high-level/gpu.md",
        ],
        "Low-Level Interface" => [
            "MRI Operators" => "low-level/operators.md",
            "Custom Reconstruction" => "low-level/custom_reconstruction.md",
            "AbstractOperators.jl" => "low-level/abstract_operators.md",
            "ProximalOperators.jl" => "low-level/proximal_operators.md",
        ],
        "Theoretical Background" => "theory.md",
        "Related packages" => "related_packages.md",
    ],
    checkdocs = :none,
    doctest = false,
    # A partial tutorial build leaves the links to the other tutorials dangling.
    warnonly = get(ENV, "RISTRETTO_DOCS_TUTORIALS", "all") == "all" ? Symbol[] : [:cross_references],
)

deploydocs(
    repo = "github.com/hakkelt/Ristretto.jl.git"
)

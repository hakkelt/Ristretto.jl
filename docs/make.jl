using Documenter
using Literate
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
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUTORIAL_DIR = joinpath(@__DIR__, "src", "tutorials")
const UTILS = joinpath(LITERATE_DIR, "NotebookUtils.jl")

function selected_tutorials()
    all = sort(filter(f -> occursin(r"^\d\d_.*\.jl$", f), readdir(LITERATE_DIR)))
    sel = get(ENV, "RISTRETTO_DOCS_TUTORIALS", "all")
    sel == "all" && return all
    sel == "none" && return String[]
    keep = split(sel, ',')
    return filter(f -> any(k -> startswith(f, lpad(k, 2, '0')), keep), all)
end

# The page loads the helper module from its source; the notebook expects it next to itself.
function page_source(content, name)
    content = replace(content, "include(\"NotebookUtils.jl\")" => "include($(repr(UTILS))) #hide")
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

function build_tutorials()
    mkpath(TUTORIAL_DIR)
    cp(UTILS, joinpath(TUTORIAL_DIR, "NotebookUtils.jl"); force = true)
    pages = Pair{String, String}[]
    for file in selected_tutorials()
        name = splitext(file)[1]
        path = joinpath(LITERATE_DIR, file)
        Literate.markdown(
            path, TUTORIAL_DIR; documenter = true, credit = false, preprocess = c -> page_source(c, name), postprocess = documenter_math,
        )
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
            "Named Dimensions" => "high-level/nameddims.md",
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

using Documenter, ProximalOperators, ProximalCore

DocMeta.setdocmeta!(ProximalCore, :DocTestSetup, :(import ProximalCore); recursive=true)

makedocs(
    modules = [ProximalOperators, ProximalCore],
    sitename = "ProximalOperators.jl",
    pages = [
        "Home" => "index.md",
        "Functions" => "functions.md",
        "Calculus rules" => "calculus.md",
        "Prox and gradient" => "operators.md",
        "Multithreading" => "threading.md",
        "GPU support" => "gpu.md",
        "Demos" => "demos.md"
    ],
    checkdocs=:none,
)

# A fork deploys to its own GitHub Pages; its workflow names the branch to deploy as `dev`.
deploydocs(
    repo = "github.com/" * get(ENV, "GITHUB_REPOSITORY", "JuliaFirstOrder/ProximalOperators.jl") * ".git",
    devbranch = get(ENV, "DOCUMENTER_DEVBRANCH", "master"),
)

using Documenter, MomentDistances

DocMeta.setdocmeta!(MomentDistances, :DocTestSetup, :(using MomentDistances); recursive=true)

makedocs(
    modules = [MomentDistances],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Tamas K. Papp",
    sitename = "MomentDistances.jl",
    pages = Any["index.md"],
    # strict = true,
    # clean = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/tpapp/MomentDistances.jl.git",
    push_preview = true
)

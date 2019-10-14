using Documenter, MLGoodies

DocMeta.setdocmeta!(MLGoodies, :DocTestSetup, quote 
    using MLGoodies
end; recursive=true)

makedocs(;
    modules=[MLGoodies],
    format=Documenter.HTML(assets=String[], prettyurls = false),
    pages=[
        "Home" => "index.md",
        "API" => [
            "Datasets" => "datasets.md",
            "Evaluation" => "evaluation.md",
            "Transforms" => "transforms.md",
            "Utils" => "utils.md",
        ]
    ],
    repo="https://github.com/dhonza/MLGoodies.jl/blob/{commit}{path}#L{line}",
    sitename="MLGoodies.jl",
    authors="Jan Drchal"
)

deploydocs(;
    repo="github.com/dhonza/MLGoodies.jl"
)

module Evaluation

using DataFrames
using DataStructures: OrderedDict
using MLDataPattern: nobs, getobs
using Statistics

using ..Transforms: ColTrans

export prob2class, probs2class, ids2classes, accuracy, cmatrix

# binary for single output
prob2class(data::AbstractMatrix{<:AbstractFloat}) = reshape(Int.(getobs(data)), :)

probs2class(data::AbstractMatrix{<:AbstractFloat}) = reshape(getindex.(argmax(data; dims=1), 1), :)

ids2classes(is::Symbol...) = begin pairs = split.(string.(is), "_"); Symbol(pairs[1][1]) => [string(v[2]) for v in pairs] end
ids2classes(i::ColTrans) = ids2classes(outids(i)...)

function accuracy(Y::AbstractVector{S}, T::AbstractVector{S}) where S
    size(Y) == size(T) || error("different dimensions size(Y)=$(size(Y)), size(T)=$(size(T))")
    mean(Y .== T)
end

accuracy(Y::AbstractMatrix{<:AbstractFloat}, T::AbstractMatrix{<:AbstractFloat}) = accuracy(probs2class(Y), probs2class(T))

function cmatrix(Y::AbstractVector{S}, T::AbstractVector{S}, classnames::Union{AbstractVector,Nothing} = nothing) where S
    size(Y) == size(T) || error("different dimensions size(Y)=$(size(Y)), size(T)=$(size(T))")
    if isnothing(classnames)
        classnames = unique(vcat(unique(Y), unique(T)))
        sort!(classnames)        
    end
    cndict = OrderedDict((cn => i for (i, cn) in enumerate(classnames))...)

    cm = zeros(Int, length(classnames), length(classnames))
    for (gnd, pred) in zip(T, Y)
         cm[cndict[gnd], cndict[pred]] += 1
    end
    names = [Symbol("truth"), (Symbol("$n (pred)") for n in classnames)...]
    DataFrame([classnames, (col for col in eachcol(cm))...], names)
end

cmatrix(Y::AbstractMatrix{<:AbstractFloat}, T::AbstractMatrix{<:AbstractFloat}, classnames::Union{AbstractVector,Nothing} = nothing) = 
    cmatrix(probs2class(Y), probs2class(T), classnames)

end
module Batches

export batchsplit, sequential_batch_producer, random_batch_producer

using Base.Iterators
using Logging
using MLDataUtils

"""
    batchsplit(n, bsize)

Return number of batches and (possibly fixed) batch size based on a number of samples `n` and target batch size `bsize`.
"""

function batchsplit(n, bsize)
    nbatches = Int(ceil(n / bsize))
    if nbatches == 1
        bsize = n
    end
    nbatches, bsize
end

"""
    sequential_batch_producer(data...; bsize, epochs = 1, func = [identity, ..], log = false)

Create a channel producing batches of maximal size `bsize` based on `data` for the number of `epochs`. 

Function `func` functions might be applied to each batch. Batches are created sequentially. Last batch 
might be shorter then all the preceding ones.
"""
function sequential_batch_producer(data...; bsize, epochs = 1, func = [identity for _ in 1:length(data)], log = false)
    @assert length(data) > 0
    @assert (func isa Vector{<:Function}) || (func isa Function)
    if func isa Function
        func = [func for _ in 1:length(data)]
    end
    @assert length(data) == length(func)
    @assert length(unique(nobs.(data))) == 1
    n = nobs(data[1])
    
    nbatches, bsize = batchsplit(n, bsize)
    
    log && @info "sequential_batch_producer for $n x $epochs = $(epochs*n) samples, $nbatches batches per epoch"
    nbatches = 0
    Channel() do channel
        for _ in 1:epochs
            last = 0
            while last < n
                range = last+1:min(last+bsize, n)
    #             @info range, length(range)
                last += length(range)
                nbatches += 1
                if all(func .== identity)
                    d = tuple((view(X, (Colon() for _ in 1:ndims(X)-1)..., range) for X in data)...)
                else
                    d = tuple((f(X[(Colon() for _ in 1:ndims(X)-1)..., range]) for (f, X) in zip(func, data))...)
                end
                put!(channel, d)
            end
        end
    log && @info "sequential_batch_producer processed $nbatches batches"
    end
end

"""
    random_batch_producer(data...; bsize, epochs = 1, func = identity, log = false)

    Create a channel producing batches of size `bsize` based on `data` for the number of `epochs`. 

Function `func` might be applied to each batch. Batches are sampled randomly (with repetition). When the number of instances `n` 
is lower than `bsize`, batch size is reduced to `n`.
"""
function random_batch_producer(data...; bsize, epochs = 1, func = identity, log = false)
    @assert length(data) > 0
    @assert length(unique(nobs.(data))) == 1
    n = nobs(data[1])
    
    nbatches, bsize = batchsplit(n, bsize)
    
    data_ = flatten(repeated(RandomBatches(data, bsize, nbatches), epochs))
    log && @info "random_batch_producer for $n x $epochs = $(epochs*n) samples, $nbatches batches per epoch"
    Channel() do channel
        nbatches = 0
        for batch in data_
            put!(channel, func.(getobs.(batch)))
            nbatches += 1
        end
    end
end

end
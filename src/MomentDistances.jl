module MomentDistances

using ArgCheck: @argcheck
using LinearAlgebra: LinearAlgebra
using Statistics: mean
using UnPack: @unpack

function distance end

struct NamedSum{T <: NamedTuple}
    named_metrics::T
end

function distance(metric::NamedSum, x, y)
    # FIXME this is probably suboptimal because of getproperty, not a concern in practice
    mapreduce(((key, metric),) -> distance(metric, getproperty(x, key), getproperty(y, key)),
              +, pairs(metric.named_metrics))
end

struct Weighted{M,T <: Real}
    metric::M
    weight::T
    function Weighted(metric::M, weight::T) where {M, T <: Real}
        @argcheck weight > 0
        new{M,T}(metric, weight)
    end
end

function distance(metric::Weighted, x, y)
    metric.weight * distance(metric.metric, x, y)
end

struct ElementwiseMean{M}
    metric::M
end

function distance(metric::ElementwiseMean, x, y)
    @unpack metric = metric
    mean(map((x, y) -> distance(metric, x, y), x, y))
end

Base.@kwdef struct AbsoluteRelative{T <: Union{Nothing,Real},F}
    relative_adjustment::T = nothing
    norm::F = LinearAlgebra.norm
end

function distance(metric::AbsoluteRelative, x, y)
    @unpack relative_adjustment, norm = metric
    Δ = norm(x - y)
    if relative_adjustment ≡ nothing
        Δ
    else
        A = max(norm(x), norm(y))
        Δ / (relative_adjustment == Inf ? A : max(1, relative_adjustment * A))
    end
end

end # module

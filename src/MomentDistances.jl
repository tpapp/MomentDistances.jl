module MomentDistances

export distance, NamedSum, Weighted, ElementwiseMean, AbsoluteRelative, summarize

using ArgCheck: @argcheck
using LinearAlgebra: LinearAlgebra
using Base.Multimedia: MIME, @MIME_str
using Statistics: mean
using UnPack: @unpack

####
#### core functionality
####

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

####
#### summaries
####
#### FIXME the whole code below is probably a weird abuse of the multimedia I/O system,
#### revamp when summarizing for other outputs. `summary` should be fully internal, an
#### implementation detail constructing a summary, `show` methods should be defined for a
#### wrapper type?

"""
    summary([options], [mime], metric, x, y)

Return summary of how a metric was calculated, according to the specific MIME type.
"""
summary(metric, x, y) = summary(MIME("text/plain"), metric, x, y)

###
### plain text summaries
###

function summary(mime::MIME"text/plain", metric, x, y)
    summary(TextSummaryOptions(), mime, metric, x, y)
end

"Options for printing text summaries."
Base.@kwdef struct TextSummaryOptions
    sigdigits::Int = 3
end

function _dotted_repr(options, x)
    lines = split(repr(x), '\n')
    line1 = first(lines)
    length(lines) > 1 ? line1 * "…" : line1
end

function _dotted_repr(options::TextSummaryOptions, x::Real)
    repr(round(x; sigdigits = options.sigdigits))
end

function _indent(str, n = 2)
    join(map(line -> ' '^n * line, split(str, '\n')), '\n')
end

function summary(options, ::MIME"text/plain", metric, x, y)
    # this is the fallback, and also the indended summary for scalar metrics like
    # AbsoluteRelative
    x_, y_, d_ = _dotted_repr.(Ref(options), (x, y, distance(metric, x, y)))
    "‹" * x_ * " ↔ " * y_ * ": " * d_ * "›"
end

function summary(options, mime::MIME"text/plain", metric::ElementwiseMean, x, y)
    @argcheck axes(x) == axes(y)
    header = "elementwise mean distance: " * _dotted_repr(options, distance(metric, x, y))
    digits_by_axis = map(1:ndims(x)) do i
        max(length(string(firstindex(x, i))), length(string(lastindex(x, i))))
    end
    body = mapreduce(*, CartesianIndices(x), x, y) do i, x, y
        padded_index = mapreduce((d, i) -> lpad(string(i), d, ' '), (a, b) -> a * "," * b,
                                 digits_by_axis, Tuple(i))
        "\n" * _indent("[" * padded_index * "]  " *
                       summary(options, mime, metric.metric, x, y))
    end
    header * body
end

function summary(options, mime::MIME"text/plain", metric::Weighted, x, y)
    "weighted: " * _dotted_repr(options, distance(metric, x, y)) * '\n' *
        _indent(summary(options, mime, metric.metric, x, y))
end

function summary(options, mime::MIME"text/plain", metric::NamedSum, x, y)
    str = "total: " * _dotted_repr(options, distance(metric, x, y))
    for (key, metric) in pairs(metric.named_metrics)
        str *= "\n" * _indent("from $(key):\n" *
                              _indent(summary(options, mime, metric,
                                              getproperty(x, key), getproperty(y, key))))
    end
    str
end

function summarize(io, mime::MIME"text/plain", metric, x, y)
    print(io, summary(mime, metric, x, y))
end

summarize(io, metric, x, y) = summarize(io, MIME("text/plain"), metric, x, y)

summarize(metric, x, y) = summarize(stdout, metric, x, y)

end # module

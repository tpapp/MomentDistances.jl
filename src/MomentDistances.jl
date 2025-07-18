"""
$(DocStringExtensions.README)
"""
module MomentDistances

export
    # generic API
    distance, # summarize,
    # primitives
    AbsDiff, RelDiff,
    # weights
    Weighted,
    # aggregation
    PNorm, NamedPNorm

using ArgCheck: @argcheck
using DocStringExtensions: FUNCTIONNAME, SIGNATURES, TYPEDEF, DocStringExtensions
using Base.Multimedia: MIME, @MIME_str

####
#### generic API
####

"""
Abstract supertype for metrics. For code organization only, it is not necessary that a
metric is a subtype, but all metrics in this package are.
"""
abstract type AbstractMetric end

Base.broadcastable(metric::AbstractMetric) = Ref(metric)

"""
`$(FUNCTIONNAME)(metric, data, model)`

Calculate the distance (a real number) between `data` and `model` using `metric`.

Importantly, `distance` is **not a metric in the mathematical sense**, for example,
it can be asymmetric.

Distances are always finite. In practice, this means that they throw a `DomainError` for
non-finite arguments.

However, it is guaranteed that `distance(metric, x, x)` is zero for all possible
`metric` and `x` values.
"""
function distance end

####
#### scalar metrics
####

struct AbsDiff <: AbstractMetric
    @doc """
    $(SIGNATURES)

    *Absolute* difference between the arguments.

    ```jldoctest
    julia> distance(AbsDiff(), 0.3, 0.6)
    0.3
    ```
    """
    AbsDiff() = new()
end

function distance(metric::AbsDiff, data::Real, model::Real)
    @argcheck isfinite(data) DomainError
    @argcheck isfinite(model) DomainError
    abs(data - model)
end

struct RelDiff <: AbstractMetric
    @doc """
    $(SIGNATURES)

    *Relative* difference between `data` and `model`, calculated as `abs(data - model) / abs(data)`.

    If `data == 0`, an error is thrown.

    ```jldoctest
    julia> distance(RelDiff(), 0.2, 0.23)
    0.15
    ```
    """
    RelDiff() = new()
end

function distance(metric::RelDiff, data::Real, model::Real)
    @argcheck isfinite(data) DomainError
    @argcheck isfinite(model) DomainError
    @argcheck data ≠ 0 DomainError
    abs(data - model) / abs(data)
end

####
#### weights
####

struct Weighted{M,T <: Real} <: AbstractMetric
    parent::M
    weight::T
    @doc """
    $(SIGNATURES)

    Multiply the distance by a positive weight.

    ```jldoctest
    julia> distance(AbsDiff(), 0.1, 0.2)
    0.1

    julia> distance(Weighted(AbsDiff(), 10), 0.1, 0.2)
    1.0
    ```

    Note that you can also create weighted metric using `weight * metric`.
    """
    function Weighted(parent::M, weight::T) where {M, T <: Real}
        @argcheck weight > 0
        new{M,T}(parent, weight)
    end
end

function Base.show(io::IO, metric::Weighted{M}) where M
    (; parent, weight) = metric
    if M <: AbstractMetric
        print(io, weight, " * ", parent)
    else
        print(io, "Weighted(", metric.metric, ", ", metric.weight, ")")
    end
end

Weighted(metric::Weighted, weight::Real) = Weighted(metric.parent, metric.weight * weight)

Base.:*(metric::AbstractMetric, weight::Real) = Weighted(metric, weight)

Base.:*(weight::Real, metric::AbstractMetric) = Weighted(metric, weight)

function distance(metric::Weighted, data, model)
    (; parent, weight) = metric
    weight * distance(parent, data, model)
end

####
#### aggregates
####

"The default `p` for p-norms is `2` (Euclidean norm)."
const DEFAULT_P = 2

struct NamedPNorm{M <: NamedTuple,T<:Real} <: AbstractMetric
    named_metrics::M
    p::T
    @doc """
    $(SIGNATURES)

    Apply the metrics in the given `NamedTuple` to `data` and `moment`, which should
    support `getproperty`.

    $(@doc DEFAULT_P)

    Note that extra properties in the arguments are ignored.
    ```jldoctest
    julia> metric = NamedPNorm((a = RelDiff(), b = AbsDiff()))
    NamedPNorm((a = RelDiff(), b = AbsDiff()))

    julia> distance(metric, (a = 1, b = 2), (a = 3, b = 4)) # ≈ √8
    2.8284271247461903

    julia> distance(metric,
           (a = 1, b = 2, c = "a fish"),     # c ignored, not in metric
           (a = 3, b = 4, d = "an octopus")) # d ignored, not in metric
    2.8284271247461903
    ```
    """
    function NamedPNorm(named_metrics::M, p::T = DEFAULT_P) where {T<:Real,M<:NamedTuple}
        @argcheck p ≥ 1
        new{M,T}(named_metrics, p)
    end
end

"""
$(SIGNATURES)

Alternative convenience constructor using keyword arguments.

```jldoctest
julia> NamedPNorm(a = RelDiff(), b = AbsDiff())
NamedPNorm((a = RelDiff(), b = AbsDiff()))
```

$(@doc DEFAULT_P)
"""
NamedPNorm(p = 2; kw...) = NamedPNorm(NamedTuple(kw), p)

function Base.show(io::IO, metric::NamedPNorm)
    p = getfield(metric, :p)
    named_metrics = getfield(metric, :named_metrics)
    print(io, "NamedPNorm(", named_metrics)
    if p ≠ DEFAULT_P
        print(io, ", ", p)
    end
    print(io, ")")
end

function Base.getproperty(metric::NamedPNorm, key::Symbol)
    getproperty(getfield(metric, :named_metrics), key)
end

"""
$(SIGNATURES)

Helper function for named sums.
"""
function _named_distance_psum(named_metrics::NamedTuple{K}, p, x, y) where K
    if @generated
        mapreduce(k -> :(abs(distance(named_metrics.$(k), x.$(k), y.$(k)))^p),
                  (a, b) -> :($(a) + $(b)), K)
    else
        mapreduce((k, v) -> abs(distance(v, getproperty(x, k), getproperty(y, k)))^p,
                  +, pairs(named_metrics))
    end
end

function distance(metric::NamedPNorm, x, y)
    p = getfield(metric, :p)
    _named_distance_psum(getfield(metric, :named_metrics), p, x, y)^(1/p)
end

struct PNorm{M,T} <: AbstractMetric
    elementwise_metric::M
    p::T
    @doc """
    $(SIGNATURES)

    Apply the elementwise metric, then calculate a p-norm.
    """
    function PNorm(elementwise_metric::M, p::T = DEFAULT_P) where {M,T}
        @argcheck p ≥ 1
        new{M,T}(elementwise_metric, p)
    end
end

function Base.show(io::IO, metric::PNorm)
    (; elementwise_metric, p) = metric
    print(io, "PNorm(", elementwise_metric)
    if p ≠ DEFAULT_P
        print(io, ", ", p)
    end
    print(io, ")")
end

function distance(metric::PNorm, data, model)
    @argcheck axes(data) == axes(model) DimensionMismatch
    (; p, elementwise_metric) = metric
    mapreduce((x, y) -> abs(distance(elementwise_metric, x, y))^p, +, data, model)^(1/p)
end


####
#### summaries
####
#### FIXME the whole code below is probably a weird abuse of the multimedia I/O system,
#### revamp when summarizing for other outputs. `summary` should be fully internal, an
#### implementation detail constructing a summary, `show` methods should be defined for a
#### wrapper type?

# """
#     summary([options], [mime], metric, x, y)

# Return summary of how a metric was calculated, according to the specific MIME type.
# """
# summary(metric, x, y) = summary(MIME("text/plain"), metric, x, y)

###
### plain text summaries
###

# function summary(mime::MIME"text/plain", metric, x, y)
#     summary(TextSummaryOptions(), mime, metric, x, y)
# end

# "Options for printing text summaries."
# Base.@kwdef struct TextSummaryOptions
#     sigdigits::Int = 3
# end

# function _dotted_repr(options, x)
#     lines = split(repr(x), '\n')
#     line1 = first(lines)
#     length(lines) > 1 ? line1 * "…" : line1
# end

# function _dotted_repr(options::TextSummaryOptions, x::Real)
#     repr(round(x; sigdigits = options.sigdigits))
# end

# function _indent(str, n = 2)
#     join(map(line -> ' '^n * line, split(str, '\n')), '\n')
# end

# function summary(options, ::MIME"text/plain", metric, x, y)
#     # this is the fallback, and also the indended summary for scalar metrics like
#     # AbsoluteRelative
#     x_, y_, d_ = _dotted_repr.(Ref(options), (x, y, distance(metric, x, y)))
#     "‹" * x_ * " ↔ " * y_ * ": " * d_ * "›"
# end

# function summary(options, mime::MIME"text/plain", metric::ElementwiseMean, x, y)
#     @argcheck axes(x) == axes(y)
#     header = "elementwise mean distance: " * _dotted_repr(options, distance(metric, x, y))
#     digits_by_axis = ntuple(ndims(x)) do i
#         max(length(string(firstindex(x, i))), length(string(lastindex(x, i))))
#     end
#     body = mapreduce(*, CartesianIndices(x), x, y) do i, x, y
#         padded_index = mapreduce((d, i) -> lpad(string(i), d, ' '), (a, b) -> a * "," * b,
#                                  digits_by_axis, Tuple(i))
#         "\n" * _indent("[" * padded_index * "]  " *
#                        summary(options, mime, metric.elementwise_metric, x, y))
#     end
#     header * body
# end

# function summary(options, mime::MIME"text/plain", metric::Weighted, x, y)
#     "weighted: " * _dotted_repr(options, distance(metric, x, y)) * '\n' *
#         _indent(summary(options, mime, metric.metric, x, y))
# end

# function summary(options, mime::MIME"text/plain", metric::NamedPNorm, x, y)
#     str = "total: " * _dotted_repr(options, distance(metric, x, y))
#     for (key, metric) in pairs(metric.named_metrics)
#         str *= "\n" * _indent("from $(key):\n" *
#                               _indent(summary(options, mime, metric,
#                                               getproperty(x, key), getproperty(y, key))))
#     end
#     str
# end

# """
# $(SIGNATURES)
# """
# function summarize(io, mime::MIME"text/plain", metric, x, y)
#     print(io, summary(mime, metric, x, y))
# end

# summarize(io, metric, x, y) = summarize(io, MIME("text/plain"), metric, x, y)

# summarize(metric, x, y) = summarize(stdout, metric, x, y)

end # module

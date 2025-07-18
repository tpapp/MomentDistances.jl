# MomentDistances

This is a Julia package to help define distance metrics, primarily for [indirect inference](https://en.wikipedia.org/wiki/Indirect_inference). Most of the functionality is trivial, in the sense that one could just as easily code it up in a few lines, but the advantage of having it in a package is that it can be documented and tested.

## Generic interface

Distances are always calculated between two arguments, conventionally named `data` and `model`. Generally, they would contain collections of the same moments calculated from the data, and simulated from a model (with given parameters).

```@docs
distance
```

## Defining metrics

Generally, one would define metric as transformations of primitives, organized into named tuples. An example would be
```@example
using MomentDistances
metric = NamedPNorm(a = RelDiff(), b = 0.3 * AbsDiff())
```
which 
1. takes named tuples,
2. compares their `a` elements using [RelDiff](@ref),
3. compares their `b` elements using [AbsDiff](@ref), multiplying it with the given weight.

### Primitives

Distance metrics are built up from *primitives*, defined on scalars.

```@docs
AbsDiff
RelDiff
```

### Weights

```@docs
Weighted
```

### Aggregation

```@docs
PNorm
NamedPNorm
```

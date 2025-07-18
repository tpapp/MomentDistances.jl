using MomentDistances
using Test
using LinearAlgebra: norm

@testset "primitives" begin
    @test @inferred(distance(AbsDiff(), 0.2, 0.3)) ≈ 0.1
    @test @inferred(distance(AbsDiff(), 0.2, 0.2)) == 0
    @test_throws DomainError distance(AbsDiff(), 0, NaN)

    @test @inferred(distance(RelDiff(), -0.2, 0.3)) ≈ 2.5
    @test_throws DomainError distance(RelDiff(), 0, 0.3)
    @test_throws DomainError distance(RelDiff(), -Inf, 0.3)
end

@testset "weighted" begin
    m = RelDiff()
    data = 0.4
    model = 0.8
    w = 0.3
    wm = Weighted(m, 0.3)
    @test repr(wm) == "$(w) * $(m)"
    @test wm == @inferred(m * w) == @inferred(w * m)
    @test @inferred(distance(wm, data, model)) ≈ w * distance(m, data, model)
    @test wm * 1 == wm
    @test @inferred(0.3 * wm * 0.4) == Weighted(m, w * 0.3 * 0.4) # recursively lift inner metric
end

@testset "pnorm" begin
    m = AbsDiff()
    p = 3.1
    pm = PNorm(m, 3.1)
    @test repr(pm) == "PNorm($m, $p)"
    N = 20
    data = randn(N)
    model = randn(N)
    @test @inferred(distance(pm, data, model)) ≈ norm(distance.(m, data, model), p)

    # invalid p
    @test_throws ArgumentError PNorm(m, 0.7)
    # default p printing
    @test repr(PNorm(m)) == "PNorm($m)"
    @test_throws DimensionMismatch distance(pm, data, randn(2 * N))
    # using * for weights
    w = 0.3
    @test w * pm == Weighted(pm, w) == pm * w
end

@testset "named pnorm" begin
    a = AbsDiff()
    b = RelDiff()
    p = 2.7
    ab = (; a, b)
    pm = NamedPNorm(ab, p)
    @test repr(pm) == "NamedPNorm($ab, $p)"
    @test NamedPNorm(p; ab...) == pm
    ad, am, bd, bm = randn(4)
    data = (; a = ad, b = bd)
    @test @inferred(distance(pm, data, (; a = am, b = bm))) ≈
        norm([distance(a, ad, am), distance(b, bd, bm)], p)

    # invalid p
    @test_throws ArgumentError NamedPNorm(ab, 0.5)
    # printing with default p
    @test repr(NamedPNorm(ab)) == "NamedPNorm($ab)"
    # ignoring extra fields
    @test @inferred(distance(pm, (; c = "ignored", data...), (; d = "ignored", data...))) == 0
    # using * for weights
    w = 0.3
    @test w * pm == Weighted(pm, w) == pm * w
end

# @testset "text summaries" begin
#     s1 = 1
#     s2 = 2
#     ms = AbsoluteRelative()
#     A1 = ones(2, 2)
#     A2 = reshape(0:3, 2, 2)
#     mA = ElementwiseMean(AbsoluteRelative())

#     @test MomentDistances.summary(ms, s1, s2) == "‹1.0 ↔ 2.0: 1.0›"
#     @test MomentDistances.summary(mA, A1, A2) ==
#         """
#         elementwise mean distance: 1.0
#           [1,1]  ‹1.0 ↔ 0.0: 1.0›
#           [2,1]  ‹1.0 ↔ 1.0: 0.0›
#           [1,2]  ‹1.0 ↔ 2.0: 1.0›
#           [2,2]  ‹1.0 ↔ 3.0: 2.0›"""
#     @test MomentDistances.summary(Weighted(mA, 0.7), A1, A2) ==
#         """
#         weighted: 0.7
#           elementwise mean distance: 1.0
#             [1,1]  ‹1.0 ↔ 0.0: 1.0›
#             [2,1]  ‹1.0 ↔ 1.0: 0.0›
#             [1,2]  ‹1.0 ↔ 2.0: 1.0›
#             [2,2]  ‹1.0 ↔ 3.0: 2.0›"""
#     @test MomentDistances.summary(NamedSum((s = ms, A = Weighted(mA, 0.7))),
#                                   (s = s1, A = A1), (s = s2, A = A2)) ==
#         """
#         total: 1.7
#           from s:
#             ‹1.0 ↔ 2.0: 1.0›
#           from A:
#             weighted: 0.7
#               elementwise mean distance: 1.0
#                 [1,1]  ‹1.0 ↔ 0.0: 1.0›
#                 [2,1]  ‹1.0 ↔ 1.0: 0.0›
#                 [1,2]  ‹1.0 ↔ 2.0: 1.0›
#                 [2,2]  ‹1.0 ↔ 3.0: 2.0›"""

#     @test sprint(summarize, ms, s1, s2) == MomentDistances.summary(ms, s1, s2)
# end

# automated AQ
import JET
JET.report_package("MomentDistances")
import Aqua
Aqua.test_all(MomentDistances)

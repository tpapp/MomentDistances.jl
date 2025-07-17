using MomentDistances
using Statistics: mean
using Test

@testset "primitives" begin
    @test distance(AbsDiff(), 0.2, 0.3) ≈ 0.1
    @test distance(AbsDiff(), 0.2, 0.2) == 0
    @test_throws DomainError distance(AbsDiff(), 0, NaN)

    @test distance(RelDiff(), 0.2, 0.3) ≈ 0.5
    @test_throws DomainError distance(RelDiff(), 0, 0.3)
    @test_throws DomainError distance(RelDiff(), -Inf, 0.3)
end

@testset "distance checks" begin
    s1 = rand() * 2
    s2 = rand() * 2

    A1 = rand(4, 4)
    A2 = rand(4, 4)

    for (ms, d) in ((AbsoluteRelative(), abs(s1 - s2)),
                    (AbsoluteRelative(; relative_adjustment = Inf),
                     abs(s1 - s2) / max(abs(s1), abs(s2))),
                    (AbsoluteRelative(; relative_adjustment = 0.3),
                     abs(s1 - s2) / max(1, 0.3 * max(abs(s1), abs(s2)))))
        @test iszero(@inferred(distance(ms, s1, s1)))
        ds = @inferred(distance(ms, s1, s2))
        @test ds ≈ d

        mA = ElementwiseMean(ms)
        @test iszero(@inferred(distance(mA, A1, A1)))
        dA = @inferred distance(mA, A1, A2)
        @test dA ≈ mean(distance.(Ref(ms), A1, A2))
        @test_throws DimensionMismatch distance(mA, A1, ones(3, 7))

        w = rand()
        mN = NamedSum((s = ms, A = Weighted(mA, w)))
        nt = (s = s1, A = A1)
        @test iszero(@inferred(distance(mN, nt, nt)))
        @test @inferred(distance(mN, nt, (s = s2, A = A2))) ≈
            distance(ms, s1, s2) + distance(mA, A1, A2) * w
    end
end

@testset "inference checks" begin
    @test @inferred(distance(AbsoluteRelative(), 1.0, 1.0)) == 0
    @test @inferred(distance(Weighted(AbsoluteRelative(), 1.0), 1.0, 1.0)) == 0
    @test @inferred(distance(ElementwiseMean(AbsoluteRelative()), ones(3), ones(3))) == 0
    metric = NamedSum((a = AbsoluteRelative(), b = Weighted(AbsoluteRelative(), 2.0),
                       c = AbsoluteRelative(), d = AbsoluteRelative()))
    nt = (a = 1, b = 2, c = 3.0, d = 4.0)
    @test @inferred(distance(metric, nt, nt)) == 0
end

@testset "constructor checks" begin
    @test_throws ArgumentError Weighted(AbsoluteRelative(), -9)
end

@testset "text summaries" begin
    s1 = 1
    s2 = 2
    ms = AbsoluteRelative()
    A1 = ones(2, 2)
    A2 = reshape(0:3, 2, 2)
    mA = ElementwiseMean(AbsoluteRelative())

    @test MomentDistances.summary(ms, s1, s2) == "‹1.0 ↔ 2.0: 1.0›"
    @test MomentDistances.summary(mA, A1, A2) ==
        """
        elementwise mean distance: 1.0
          [1,1]  ‹1.0 ↔ 0.0: 1.0›
          [2,1]  ‹1.0 ↔ 1.0: 0.0›
          [1,2]  ‹1.0 ↔ 2.0: 1.0›
          [2,2]  ‹1.0 ↔ 3.0: 2.0›"""
    @test MomentDistances.summary(Weighted(mA, 0.7), A1, A2) ==
        """
        weighted: 0.7
          elementwise mean distance: 1.0
            [1,1]  ‹1.0 ↔ 0.0: 1.0›
            [2,1]  ‹1.0 ↔ 1.0: 0.0›
            [1,2]  ‹1.0 ↔ 2.0: 1.0›
            [2,2]  ‹1.0 ↔ 3.0: 2.0›"""
    @test MomentDistances.summary(NamedSum((s = ms, A = Weighted(mA, 0.7))),
                                  (s = s1, A = A1), (s = s2, A = A2)) ==
        """
        total: 1.7
          from s:
            ‹1.0 ↔ 2.0: 1.0›
          from A:
            weighted: 0.7
              elementwise mean distance: 1.0
                [1,1]  ‹1.0 ↔ 0.0: 1.0›
                [2,1]  ‹1.0 ↔ 1.0: 0.0›
                [1,2]  ‹1.0 ↔ 2.0: 1.0›
                [2,2]  ‹1.0 ↔ 3.0: 2.0›"""

    @test sprint(summarize, ms, s1, s2) == MomentDistances.summary(ms, s1, s2)
end

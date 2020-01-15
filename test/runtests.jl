using MomentDistances: distance, AbsoluteRelative, ElementwiseMean, NamedSum, Weighted
using Statistics: mean
using Test

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
        @test iszero(distance(ms, s1, s1))
        ds = distance(ms, s1, s2)
        @test ds ≈ d

        mA = ElementwiseMean(ms)
        @test iszero(distance(mA, A1, A1))
        dA = distance(mA, A1, A2)
        @test dA ≈ mean(distance.(Ref(ms), A1, A2))
        @test_throws DimensionMismatch distance(mA, A1, ones(3, 7))

        w = rand()
        mN = NamedSum((s = ms, A = Weighted(mA, w)))
        nt = (s = s1, A = A1)
        @test iszero(distance(mN, nt, nt))
        @test distance(mN, nt, (s = s2, A = A2)) ≈ distance(ms, s1, s2) + distance(mA, A1, A2) * w
    end
end

@testset "constructor checks" begin
    @test_throws ArgumentError Weighted(AbsoluteRelative(), -9)
end

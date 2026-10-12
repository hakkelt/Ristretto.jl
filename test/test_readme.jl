using TestItems

@testitem "README examples run" tags = [:integration] begin
    using Test
    using Ristretto
    using Random: Random

    # The variables the README's examples assume: an 8-coil, 4× undersampled acquisition.
    Random.seed!(1)
    n = 64
    img_true = zeros(ComplexF32, n, n)
    img_true[16:48, 20:44] .= 1
    smaps = ComplexF32.(coil_sensitivities(n, n, 8))
    pattern = create_sampling_pattern(VariableDensitySampling(PolynomialDistribution(3), 4.0, 0.15), (n, n))
    mask = pattern[2]
    sim = simulate_acquisition(
        img_true, AcquisitionInfo(; image_size = (n, n), subsampling = pattern, sensitivity_maps = smaps);
        inverse_crime_check = false, keep_sensitivity_maps = true,
    )
    kspace = sim.kspace_data

    readme = read(joinpath(pkgdir(Ristretto), "README.md"), String)
    blocks = [m[1] for m in eachmatch(r"```julia\n(.*?)```"s, readme)]
    # The installation block would install the package; the others are the examples.
    examples = filter(b -> !occursin("Pkg.add", b), blocks)
    @test length(examples) == 2
    m = Module()
    for (name, value) in (:kspace => kspace, :smaps => smaps, :mask => mask)
        Core.eval(m, :($name = $value))
    end
    for code in examples
        Core.eval(m, Meta.parseall(code))
        img = Core.eval(m, :img)
        @test size(img)[1:2] == (n, n)
        @test all(isfinite, img)
    end
end

@testitem "NonCartesianAcquisitionInfo" tags = [:acquisition, :nfft] begin
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, NonCartesianAcquisitionInfo
    using NamedDims

    @testset "Basic 2D construction" begin
        trajectory = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8))
        @test info isa NonCartesianAcquisitionInfo
        @test info.is3D == false
        @test info.image_size == (8, 8)
        @test info.trajectory === trajectory
        @test isnothing(info.dcf)
        @test isnothing(info.sensitivity_maps)
    end

    @testset "3D construction" begin
        trajectory = randn(Float32, 3, 128)
        ksp = randn(ComplexF32, 128)
        info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8, 8))
        @test info isa NonCartesianAcquisitionInfo
        @test info.is3D == true
        @test info.image_size == (8, 8, 8)
    end

    @testset "With DCF" begin
        trajectory = randn(Float32, 2, 64)
        dcf = rand(Float32, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, dcf, image_size = (8, 8))
        @test info.dcf === dcf
    end

    @testset "With sensitivity maps" begin
        trajectory = randn(Float32, 2, 64)
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp_coil = NamedDimsArray(randn(ComplexF32, 64, 4), (:sample, :coil))
        traj_named = NamedDimsArray(trajectory, (:coord, :sample))
        info = AcquisitionInfo(ksp_coil; trajectory = traj_named, sensitivity_maps = smaps, image_size = (8, 8))
        @test info.sensitivity_maps === smaps
    end

    @testset "NamedDims trajectory" begin
        trajectory = NamedDimsArray(randn(Float32, 2, 64), (:dim, :sample))
        ksp = NamedDimsArray(randn(ComplexF32, 64), (:sample,))
        info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8))
        @test info isa NonCartesianAcquisitionInfo
    end

    @testset "Validation errors" begin
        # Bad trajectory ndims
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory = randn(Float32, 3), image_size = (8, 8))
        # Wrong coordinate dimension
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory = randn(Float32, 4, 64), image_size = (8, 8, 8))
        # Mismatched image_size
        @test_throws ArgumentError AcquisitionInfo(nothing; trajectory = randn(Float32, 2, 64), image_size = (8, 8, 8))
        # DCF type mismatch
        @test_throws ArgumentError AcquisitionInfo(
            randn(ComplexF32, 64);
            trajectory = randn(Float32, 2, 64),
            dcf = rand(Float64, 64),
            image_size = (8, 8),
        )
    end

    @testset "Copy constructor" begin
        trajectory = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64)
        info = AcquisitionInfo(ksp; trajectory, image_size = (8, 8))

        new_ksp = randn(ComplexF32, 64)
        info2 = AcquisitionInfo(info; kspace_data = new_ksp)
        @test info2.kspace_data === new_ksp
        @test info2.trajectory === trajectory
        @test info2 isa NonCartesianAcquisitionInfo
    end
end

@testitem "AcquisitionInfo copy constructors field round-trips" tags = [:acquisition] begin
    using Ristretto: CartesianAcquisitionInfo
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator, NonCartesianAcquisitionInfo

    @testset "CartesianAcquisitionInfo individual field round-trips" begin
        mask = rand(Bool, 16, 16)
        ksp = randn(ComplexF32, sum(mask), 4)
        smaps = randn(ComplexF32, 16, 16, 4)
        orig = CartesianAcquisitionInfo(
            ksp;
            is3D = false,
            image_size = (16, 16),
            sensitivity_maps = smaps,
            subsampling = mask,
            shifted_kspace_dims = (1,),
            shifted_image_dims = (2,),
        )

        # Override kspace_data
        new_ksp = randn(ComplexF32, sum(mask), 4)
        c1 = AcquisitionInfo(orig; kspace_data = new_ksp)
        @test c1 isa CartesianAcquisitionInfo
        @test c1.kspace_data === new_ksp
        @test c1.is3D === orig.is3D
        @test c1.image_size === orig.image_size
        @test c1.sensitivity_maps === orig.sensitivity_maps
        @test c1.subsampling === orig.subsampling
        @test c1.shifted_kspace_dims === orig.shifted_kspace_dims
        @test c1.shifted_image_dims === orig.shifted_image_dims

        # Override sensitivity_maps
        new_smaps = randn(ComplexF32, 16, 16, 4)
        c2 = CartesianAcquisitionInfo(orig; sensitivity_maps = new_smaps)
        @test c2.sensitivity_maps === new_smaps
        @test c2.kspace_data === orig.kspace_data
        @test c2.image_size === orig.image_size
        @test c2.subsampling === orig.subsampling

        # Override image_size
        plain = CartesianAcquisitionInfo(ksp; is3D = false, image_size = (16, 16))
        c3 = AcquisitionInfo(plain; image_size = (32, 32))
        @test c3.image_size == (32, 32)
        @test c3.kspace_data === plain.kspace_data
        @test c3.is3D === plain.is3D

        # Override subsampling
        new_mask = (rand(Bool, 16, 16),)
        c4 = AcquisitionInfo(orig; subsampling = new_mask)
        @test c4.subsampling === new_mask
        @test c4.kspace_data === orig.kspace_data

        # Override shifted dims
        c5 = AcquisitionInfo(orig; shifted_kspace_dims = (2,), shifted_image_dims = (1,))
        @test c5.shifted_kspace_dims == (2,)
        @test c5.shifted_image_dims == (1,)
        @test c5.kspace_data === orig.kspace_data
    end

    @testset "NonCartesianAcquisitionInfo individual field round-trips" begin
        ksp = randn(ComplexF32, 64, 4)
        traj = randn(Float32, 2, 64)
        dcf = rand(Float32, 64)
        smaps = randn(ComplexF32, 16, 16, 4)
        orig = NonCartesianAcquisitionInfo(
            ksp;
            trajectory = traj,
            dcf = dcf,
            sensitivity_maps = smaps,
            image_size = (16, 16),
            shifted_kspace_dims = (1,),
            shifted_image_dims = (2,),
        )
        @test !orig.is3D

        # Override kspace_data
        new_ksp = randn(ComplexF32, 64, 4)
        c1 = AcquisitionInfo(orig; kspace_data = new_ksp)
        @test c1 isa NonCartesianAcquisitionInfo
        @test c1.kspace_data === new_ksp
        @test c1.trajectory === orig.trajectory
        @test c1.dcf === orig.dcf
        @test c1.sensitivity_maps === orig.sensitivity_maps
        @test c1.image_size === orig.image_size
        @test c1.is3D === orig.is3D
        @test c1.shifted_kspace_dims === orig.shifted_kspace_dims
        @test c1.shifted_image_dims === orig.shifted_image_dims

        # Override trajectory (2D to 3D)
        traj3d = randn(Float32, 3, 128)
        ksp3d = randn(ComplexF32, 128, 4)
        dcf3d = rand(Float32, 128)
        smaps3d = randn(ComplexF32, 16, 16, 16, 4)
        c2 = NonCartesianAcquisitionInfo(
            orig;
            trajectory = traj3d,
            kspace_data = ksp3d,
            dcf = dcf3d,
            sensitivity_maps = smaps3d,
            image_size = (16, 16, 16),
        )
        @test c2.is3D == true
        @test c2.trajectory === traj3d
        @test c2.image_size == (16, 16, 16)

        # Override dcf
        new_dcf = rand(Float32, 64)
        c3 = AcquisitionInfo(orig; dcf = new_dcf)
        @test c3.dcf === new_dcf
        @test c3.kspace_data === orig.kspace_data
        @test c3.trajectory === orig.trajectory
        @test c3.sensitivity_maps === orig.sensitivity_maps

        # Override sensitivity_maps
        new_smaps = randn(ComplexF32, 16, 16, 4)
        c4 = AcquisitionInfo(orig; sensitivity_maps = new_smaps)
        @test c4.sensitivity_maps === new_smaps
        @test c4.kspace_data === orig.kspace_data
        @test c4.trajectory === orig.trajectory

        # Override image_size
        plain_nc = NonCartesianAcquisitionInfo(ksp; trajectory = traj, image_size = (16, 16))
        c5 = AcquisitionInfo(plain_nc; image_size = (32, 32))
        @test c5.image_size == (32, 32)
        @test c5.kspace_data === plain_nc.kspace_data
        @test c5.trajectory === plain_nc.trajectory
    end
end

@testitem "Dimension utilities" tags = [:acquisition] begin
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator
    using NamedDims
    import Ristretto: get_image_size, get_time_dim,
        get_fourier_kspace_dims, get_fourier_image_dims,
        get_nonfourier_image_dims, get_nonfourier_kspace_dims, get_image_dims

    @testset "get_image_size - 2D Cartesian" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_image_size(info) == (8, 8)
    end

    @testset "get_image_size - 2D Cartesian with batch dims" begin
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_image_size(info) == (8, 8, 5)
    end

    @testset "get_image_size - 2D+coil with batch dims" begin
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp = randn(ComplexF32, 8, 8, 4, 5)
        info = AcquisitionInfo(ksp; image_size = (8, 8), sensitivity_maps = smaps)
        @test get_image_size(info) == (8, 8, 5)
    end

    @testset "get_time_dim" begin
        @test get_time_dim(3, 1:4) == 3
        @test get_time_dim(:time, (:x, :y, :time)) == 3
        @test_throws ArgumentError get_time_dim(nothing, 1:4)  # no NamedDims, no time_dim
        @test_throws ArgumentError get_time_dim(nothing, (:x, :y, :z))  # no :time dimension
    end

    @testset "get_fourier_kspace_dims - 2D" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_fourier_kspace_dims(info) == 1:2
    end

    @testset "get_fourier_kspace_dims - 3D" begin
        ksp = randn(ComplexF32, 8, 8, 8)
        info = AcquisitionInfo(ksp; image_size = (8, 8, 8), is3D = true)
        @test get_fourier_kspace_dims(info) == 1:3
    end

    @testset "get_fourier_kspace_dims - NamedDims" begin
        ksp = NamedDimsArray(randn(ComplexF32, 8, 8), (:kx, :ky))
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_fourier_kspace_dims(info) == (:kx, :ky)
    end

    @testset "get_fourier_image_dims" begin
        ksp = randn(ComplexF32, 8, 8)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_fourier_image_dims(info) == 1:2

        ksp3d = randn(ComplexF32, 8, 8, 8)
        info3d = AcquisitionInfo(ksp3d; image_size = (8, 8, 8), is3D = true)
        @test get_fourier_image_dims(info3d) == 1:3
    end

    @testset "get_nonfourier_kspace_dims" begin
        # 2D with batch dimension
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_nonfourier_kspace_dims(info) == 3:3

        # 2D with coils and batch dimension
        smaps = randn(ComplexF32, 8, 8, 4)
        ksp_coil = randn(ComplexF32, 8, 8, 4, 5)
        info_coil = AcquisitionInfo(ksp_coil; image_size = (8, 8), sensitivity_maps = smaps)
        @test get_nonfourier_kspace_dims(info_coil) == 4:4
    end

    @testset "get_image_dims" begin
        ksp = randn(ComplexF32, 8, 8, 5)
        info = AcquisitionInfo(ksp; image_size = (8, 8))
        @test get_image_dims(info) == 1:3
    end

    @testset "Non-Cartesian dimensions" begin
        traj = randn(Float32, 2, 64)
        ksp = randn(ComplexF32, 64, 5)
        info = AcquisitionInfo(ksp; trajectory = traj, image_size = (8, 8))
        @test get_fourier_kspace_dims(info) == 1:1  # 1 sample dim for non-cartesian
        @test get_image_size(info) == (8, 8, 5)
    end
end

@testitem "RegularLatticeSampling" tags = [:simulation] begin
    using Ristretto
    using Ristretto: check_applicable

    @testset "plain regular lattice" begin
        _, mask = create_sampling_pattern(RegularLatticeSampling(4), (128, 128))
        @test findall(mask) == collect(1:4:128)
        @test length(mask) / sum(mask) ≈ 4
    end

    @testset "ACS band" begin
        _, mask = create_sampling_pattern(RegularLatticeSampling(4; center_fraction = 0.1), (128, 128))
        acs = 58:70   # the fully sampled centre `get_fully_sampled_region` picks for cf = 0.1
        @test all(mask[acs])
        # Every lattice line is still acquired, so the net acceleration is below the nominal one.
        @test all(mask[1:4:128])
        @test 1 < length(mask) / sum(mask) < 4
    end

    @testset "acceleration is factored over two subsampled dimensions" begin
        _, mask = create_sampling_pattern(RegularLatticeSampling(4), (64, 64, 32))
        @test size(mask) == (64, 32)
        @test length(mask) / sum(mask) ≈ 4      # 2 × 2
        # 3 × 1: only the divisors of R are candidates. 64 is not a multiple of 3, so the last
        # partial stride makes the realized acceleration slightly below the nominal one.
        _, mask3 = create_sampling_pattern(RegularLatticeSampling(3), (64, 64, 32))
        @test length(mask3) / sum(mask3) ≈ 3 rtol = 0.05
    end

    @testset "deterministic" begin
        a = create_sampling_pattern(RegularLatticeSampling(3; center_fraction = 0.05), (64, 64))
        b = create_sampling_pattern(RegularLatticeSampling(3; center_fraction = 0.05), (64, 64))
        @test a[2] == b[2]
    end

    @testset "argument validation" begin
        @test_throws ArgumentError RegularLatticeSampling(2.5)
        @test_throws ArgumentError RegularLatticeSampling(0.5)
        @test_throws ArgumentError RegularLatticeSampling(2; center_fraction = 1.0)
    end

    @testset "GRAPPA accepts the generated pattern" begin
        nx, ny, nc = 64, 64, 4
        img = ComplexF32.(reshape(range(0.0f0, 1.0f0; length = nx * ny), nx, ny))
        pattern = create_sampling_pattern(RegularLatticeSampling(2; center_fraction = 0.15), (nx, ny))
        data = simulate_acquisition(
            img,
            AcquisitionInfo(;
                is3D = false, image_size = (nx, ny), subsampling = pattern,
                sensitivity_maps = coil_sensitivities(nx, ny, nc),
            ); inverse_crime_check = false, keep_sensitivity_maps = true
        )
        @test check_applicable(GRAPPA(), data) === nothing
    end
end

@testitem "PartialFourierSampling" tags = [:simulation] begin
    using Ristretto

    @testset "contiguous band from the first index" begin
        _, mask = create_sampling_pattern(PartialFourierSampling(0.7), (128, 128))
        last_acquired = round(Int, 0.7 * 128)
        @test findall(mask) == collect(1:last_acquired)
        # The acquired band covers the k-space centre, which is what makes it recoverable.
        @test any(mask[60:70])
    end

    @testset "no truncation at fraction 1" begin
        _, mask = create_sampling_pattern(PartialFourierSampling(1.0), (64, 64))
        @test all(mask)
    end

    @testset "deterministic" begin
        a = create_sampling_pattern(PartialFourierSampling(0.625), (64, 64))
        b = create_sampling_pattern(PartialFourierSampling(0.625), (64, 64))
        @test a[2] == b[2]
    end

    @testset "argument validation" begin
        @test_throws ArgumentError PartialFourierSampling(1.5)
        @test_throws ArgumentError PartialFourierSampling(0.0)
    end
end

@testitem "Sampling patterns" tags = [:simulation] begin
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator

    @testset "PoissonDiskSampling" begin
        pattern = PoissonDiskSampling(4.0)
        # PoissonDisk requires 3D dims with subsample_freq_encoding=false → returns (:, 2D mask)
        result = create_sampling_pattern(pattern, (64, 64, 64))
        @test result isa Tuple
        @test result[1] === Colon()
        mask = result[2]
        @test size(mask) == (64, 64)
        @test eltype(mask) == Bool
        @test sum(mask) < prod(size(mask))
        @test sum(mask) > 0
    end

    @testset "PoissonDiskSampling 2D with freq encoding" begin
        pattern = PoissonDiskSampling(4.0)
        # PoissonDisk with 2D + subsample_freq_encoding=true → returns plain 2D mask
        mask = create_sampling_pattern(pattern, (32, 32); subsample_freq_encoding = true)
        @test size(mask) == (32, 32)
        @test eltype(mask) == Bool
        @test sum(mask) > 0
    end

    @testset "VariableDensitySampling with Gaussian" begin
        pattern = VariableDensitySampling(GaussianDistribution(), 4.0)
        # 2D, subsample_freq_encoding=false → returns (:, 1D mask)
        result = create_sampling_pattern(pattern, (64, 64))
        @test result isa Tuple
        mask = result[2]
        @test size(mask) == (64,)
        @test eltype(mask) == Bool
        @test sum(mask) < length(mask)
    end

    @testset "VariableDensitySampling with Polynomial" begin
        pattern = VariableDensitySampling(PolynomialDistribution(), 4.0)
        result = create_sampling_pattern(pattern, (64, 64))
        @test result isa Tuple
        mask = result[2]
        @test size(mask) == (64,)
        @test eltype(mask) == Bool
    end

    @testset "to_displayable_mask" begin
        # to_displayable_mask expects (:, vector) or (:, array) or plain array
        inner_mask = BitVector(rand(Bool, 8))
        pattern = (:, inner_mask)
        displayable = to_displayable_mask(pattern, (8, 8))
        @test size(displayable) == (8, 8)
        @test eltype(displayable) == Bool

        # Every per-axis form `CartesianAcquisitionInfo` accepts: an index vector or range selects
        # the same phase encodes as the Bool vector it stands for, on any axis.
        keep = BitVector([1, 0, 1, 1, 0, 0, 1, 0])
        @test to_displayable_mask((:, findall(keep)), (8, 8)) == to_displayable_mask((:, keep), (8, 8))
        @test to_displayable_mask((:, 2:2:8), (8, 8)) == repeat(reshape(iseven.(1:8), 1, :), 8, 1)
        rows = BitVector([1, 1, 0, 0, 1, 1, 0, 0])
        @test to_displayable_mask((rows, keep), (8, 8)) == rows .& reshape(keep, 1, :)
        @test to_displayable_mask((:, [2, 5], :), (4, 6, 3)) == [j in (2, 5) for i in 1:4, j in 1:6, k in 1:3]
        @test_throws ArgumentError to_displayable_mask((:, [1, 2]), (4, 4, 4))
    end

    @testset "Anisotropic dims and large center fractions" begin
        # The fully sampled center region must be clamped to the array bounds
        pattern = UniformRandomSampling(2.0, 0.5)
        mask = create_sampling_pattern(pattern, (4, 100); subsample_freq_encoding = true)
        @test size(mask) == (4, 100)
        @test sum(mask) > 0

        # The center region may exceed the sample budget; sample count must not go negative
        pattern = VariableDensitySampling(GaussianDistribution(), 8.0, 0.9)
        result = create_sampling_pattern(pattern, (32, 32))
        @test result isa Tuple
        @test sum(result[2]) > 0

        pattern = PoissonDiskSampling(16.0, 0.9)
        mask = create_sampling_pattern(pattern, (16, 16); subsample_freq_encoding = true)
        @test sum(mask) > 0

        # PolynomialDistribution with an odd exponent used to throw "Negative weight found in
        # weight vector" at anisotropic dims/center_fraction combinations that push a grid
        # corner's normalized distance past 1 (an even exponent hid the same out-of-range value).
        for (dims, R, cf) in (((64, 64, 16), 4.0, 0.1), ((64, 64, 32), 4.0, 0.1))
            pattern = VariableDensitySampling(PolynomialDistribution(3), R, cf)
            result = create_sampling_pattern(pattern, dims)
            @test result isa Tuple
            @test sum(result[2]) > 0
        end
    end

    @testset "PoissonDiskSampling with Real arguments" begin
        pattern = PoissonDiskSampling(4, 1 // 10)
        @test pattern.acceleration == 4.0
        @test pattern.center_fraction == 0.1
    end
end

@testitem "CartesianAcquisitionInfo shifted dims" tags = [:acquisition] begin
    using Ristretto: CartesianAcquisitionInfo
    using Ristretto
    using Ristretto: get_encoding_operator, get_fourier_operator
    using NamedDims

    @testset "shifted_kspace_dims as single Integer" begin
        ksp = rand(ComplexF32, 16, 16, 2)
        acq = AcquisitionInfo(ksp; is3D = false, shifted_kspace_dims = 1)
        @test acq.shifted_kspace_dims == (1,)
    end

    @testset "shifted_kspace_dims as Symbol" begin
        ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 16, 16, 2))
        acq = AcquisitionInfo(ksp; shifted_kspace_dims = :kx)
        @test acq.shifted_kspace_dims == (:kx,)
    end

    @testset "shifted_image_dims as single Integer" begin
        ksp = rand(ComplexF32, 16, 16, 2)
        acq = AcquisitionInfo(ksp; is3D = false, shifted_image_dims = 1)
        @test acq.shifted_image_dims == (1,)
    end

    @testset "shifted_image_dims as Symbol" begin
        ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 16, 16, 2))
        acq = AcquisitionInfo(ksp; shifted_image_dims = :x)
        @test acq.shifted_image_dims == (:x,)
    end

    @testset "shifted_kspace_dims as Tuple of Integers" begin
        ksp = rand(ComplexF32, 16, 16, 2)
        acq = AcquisitionInfo(ksp; is3D = false, shifted_kspace_dims = (1, 2))
        @test acq.shifted_kspace_dims == (1, 2)
    end

    @testset "Symbol shifted dims build the same operator as Int dims" begin
        ksp = NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 16, 16))
        img = NamedDimsArray{(:x, :y)}(rand(ComplexF32, 16, 16))

        acq_sym = AcquisitionInfo(ksp; shifted_kspace_dims = :kx, shifted_image_dims = :y)
        acq_int = AcquisitionInfo(ksp; shifted_kspace_dims = 1, shifted_image_dims = 2)

        F_sym = get_fourier_operator(acq_sym)
        F_int = get_fourier_operator(acq_int)
        @test parent(F_sym * img) ≈ parent(F_int * img)

        E_sym = get_encoding_operator(acq_sym)
        @test parent(E_sym * img) ≈ parent(F_int * img)

        @test_throws ArgumentError get_fourier_operator(
            AcquisitionInfo(ksp; shifted_kspace_dims = :kz)
        )
    end

    @testset "NamedDims 2D multislice smaps validation" begin
        nx, ny, nc, nz = 8, 8, 2, 3
        ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, nx, ny, nc, nz))
        smaps = NamedDimsArray{(:x, :y, :coil, :z)}(rand(ComplexF32, nx, ny, nc, nz))
        acq = AcquisitionInfo(ksp; sensitivity_maps = smaps)
        @test !acq.is3D
        @test ndims(acq.sensitivity_maps) == 4
    end
end

@testitem "3D k-space subsampled over ky–kz with sensitivity maps" tags = [:acquisition] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using NamedDims
    using Random

    # A ky–kz mask joins the two phase-encode axes into one `:kyz` axis, so the k-space has no
    # `:kz` of its own; the maps check must accept that, and the named acquisition must
    # reconstruct exactly what the plain-array one does.
    rng = MersenneTwister(3)
    nx, ny, nz, nc = 8, 6, 4, 2
    mask = rand(rng, Bool, ny, nz)
    mask[1, 1] = true
    ksp = rand(rng, ComplexF32, nx, count(mask), nc)
    smaps = rand(rng, ComplexF32, nx, ny, nz, nc)
    plain = CartesianAcquisitionInfo(
        ksp; is3D = true, image_size = (nx, ny, nz), sensitivity_maps = smaps, subsampling = (:, mask),
    )
    named = CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :kyz, :coil)}(ksp); is3D = true, image_size = (nx, ny, nz),
        sensitivity_maps = NamedDimsArray{(:x, :y, :z, :coil)}(smaps), subsampling = (:, mask),
    )
    x_plain = reconstruct(plain, DirectReconstruction(); verbosity = Silent())
    x_named = reconstruct(named, DirectReconstruction(); verbosity = Silent())
    @test dimnames(x_named) == (:x, :y, :z)
    @test parent(x_named) ≈ x_plain

    # Without maps the coil axis of the per-coil images sits at image position 4, not at its
    # k-space position 3.
    for ksp_in in (ksp, NamedDimsArray{(:kx, :kyz, :coil)}(ksp))
        acq = CartesianAcquisitionInfo(ksp_in; is3D = true, image_size = (nx, ny, nz), subsampling = (:, mask))
        @test size(reconstruct(acq, DirectReconstruction(RootSumSquares()); verbosity = Silent())) == (nx, ny, nz)
        @test size(reconstruct(acq, DirectReconstruction(); verbosity = Silent())) == (nx, ny, nz, nc)
    end

    # A 3D k-space with no kz-carrying axis at all is still rejected.
    @test_throws ArgumentError CartesianAcquisitionInfo(
        NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, nx, ny, nc)); is3D = true,
        image_size = (nx, ny, nz), sensitivity_maps = NamedDimsArray{(:x, :y, :z, :coil)}(smaps),
    )
end

@testitem "GRAPPA on a phase-encode index vector" tags = [:acquisition, :reconstruction] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo
    using NamedDims

    # GRAPPA reads the pattern through `to_displayable_mask`, which used to reject an index vector
    # with "Unsupported pattern format" although the acquisition accepts one.
    n, nc = 32, 4
    lines = sort(unique([1:2:n; 13:20]))
    ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, n, length(lines), nc))
    method = GRAPPA(kernel_size = (4, 3), calib_size = (8, 8), coil_combination = RootSumSquares())
    by_index = CartesianAcquisitionInfo(ksp; is3D = false, image_size = (n, n), subsampling = (:, lines))
    by_mask = CartesianAcquisitionInfo(ksp; is3D = false, image_size = (n, n), subsampling = (:, in(lines).(1:n)))
    @test parent(reconstruct(by_index, method; verbosity = Silent())) ≈ parent(reconstruct(by_mask, method; verbosity = Silent()))
end

@testitem "Adapt: an acquisition moves to a device and back" tags = [:acquisition, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo, PartitionedKSpace
    using NamedDims

    ksp = NamedDimsArray{(:kx, :ky, :coil)}(rand(ComplexF32, 8, 6, 2))
    smaps = NamedDimsArray{(:x, :y, :coil)}(rand(ComplexF32, 8, 8, 2))
    mask = [isodd(i) || i in (4, 8) for i in 1:8]
    cart = CartesianAcquisitionInfo(ksp; is3D = false, image_size = (8, 8), sensitivity_maps = smaps, subsampling = (:, mask))

    traj = rand(Float32, 2, 16, 3) .- 0.5f0
    nc_ksp = rand(ComplexF32, 16, 3, 2)
    noncart = NonCartesianAcquisitionInfo(nc_ksp; trajectory = traj, image_size = (8, 8), dcf = rand(Float32, 16, 3))

    parts = PartitionedKSpace(
        [rand(ComplexF32, 8, n, 2) for n in (3, 5)]; ragged_dim = 2, dimnames = (:kx, :ky, :coil, :time)
    )

    for backend in all_backends()
        @testset "$(backend.name)" begin
            d = to_device(backend, cart)
            @test _is_device(d) && _is_device(d.sensitivity_maps)
            @test dimnames(d.kspace_data) == dimnames(ksp) && dimnames(d.sensitivity_maps) == dimnames(smaps)
            # The pattern stays on the host: the sampling operators index with host indices.
            @test d.subsampling === cart.subsampling
            back = Adapt.adapt(Array, d)
            @test !_is_device(back) && parent(back.kspace_data) == parent(ksp)

            dn = to_device(backend, noncart)
            @test _is_device(dn) && _is_device(dn.dcf)
            @test dn.trajectory === noncart.trajectory

            dp = Adapt.adapt(backend.array_type, parts)
            @test _is_device(dp) && all(_is_device, dp.parts)
            @test dimnames(first(dp.parts)) == (:kx, :ky, :coil)
        end
    end
end

@testitem "Device storage: mismatches and the settings a device run resolves to" tags = [:acquisition, :reconstruction, :gpu] setup = [GpuEnvSetup, GpuHelpers] begin
    using Ristretto
    using Ristretto: CartesianAcquisitionInfo, resolve_config, DEVICE_DISABLES_TASK_SPLITTING
    using NamedDims

    ksp = NamedDimsArray{(:kx, :ky, :coil, :z)}(rand(ComplexF32, 8, 8, 2, 3))
    smaps = NamedDimsArray{(:x, :y, :coil, :z)}(rand(ComplexF32, 8, 8, 2, 3))
    host = CartesianAcquisitionInfo(ksp; is3D = false, sensitivity_maps = smaps)

    for backend in all_backends()
        @testset "$(backend.name)" begin
            dev = to_device(backend, host)
            # The k-space and the maps must be in the same memory.
            @test_throws ArgumentError CartesianAcquisitionInfo(dev.kspace_data; is3D = false, sensitivity_maps = smaps)
            # So must an initial guess.
            @test_throws ArgumentError reconstruct(dev; x₀ = zeros(ComplexF32, 8, 8, 3), verbosity = Silent())
            # Slices spread over threads would share one device.
            @test_throws ArgumentError reconstruct(dev; task_executor = MultiThreadingExecutor(), verbosity = Silent())

            conf = resolve_config(ReconstructionConfig(; threaded = true), dev, DirectReconstruction())
            @test conf.disable_task_splitting === DEVICE_DISABLES_TASK_SPLITTING
            @test !conf.threaded
            @test resolve_config(ReconstructionConfig(; disable_task_splitting = false), dev, DirectReconstruction()).disable_task_splitting === false
        end
    end
    @test resolve_config(ReconstructionConfig(), host, DirectReconstruction()).disable_task_splitting === false
end

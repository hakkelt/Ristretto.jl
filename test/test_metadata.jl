using TestItems

@testitem "Header: known keys are fields, any other key is kept" tags = [:acquisition] begin
    using Test
    using Ristretto
    using Ristretto: header

    h = Header(; fov = (240, 180), TE = 4.2, protocol = "t1_se")
    @test h isa AbstractDict{Symbol, Any}
    @test h[:fov] === (240.0, 180.0)
    @test h.TE === 4.2
    @test h[:protocol] == "t1_se"
    @test isnothing(h.offset)
    @test !haskey(h, :offset)
    @test_throws KeyError h[:offset]
    @test get(h, :offset, 0) == 0
    @test Set(keys(h)) == Set((:fov, :TE, :protocol))
    @test length(h) == 3
    h[:offset] = [1, 2, 3]
    @test h.offset === (1.0, 2.0, 3.0)
    h.TR = 10
    @test h[:TR] === 10.0
    h[:site] = "A"
    @test h.extra[:site] == "A"
    delete!(h, :TR)
    @test isnothing(h.TR)
    @test_throws Exception h.protocl
    @test_throws ArgumentError h.protocol = "x"
    @test_throws ArgumentError Header(; orientation = rand(2, 2))
    @test_throws ArgumentError Header(; offset = (1, 2))
    @test_throws ArgumentError Header(; fov = (1,))
    @test Header(Dict("TE" => 2)) == Header(; TE = 2)
    @test occursin("fov = (240.0, 180.0)", sprint(show, MIME"text/plain"(), h))

    hc = copy(h)
    hc.tags["x"] = 1
    hc[:site] = "B"
    @test isempty(h.tags)
    @test h[:site] == "A"
end

@testitem "Header on an acquisition: optional, stored as given, shared by copies" tags = [:acquisition] begin
    using Test
    using Ristretto
    using Ristretto: header

    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false)
    @test acq.image_size == (16, 12)
    @test header(acq) isa Header
    @test isempty(header(acq))

    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (240, 180), protocol = "t1_se"))
    @test header(acq).fov == (240.0, 180.0)
    @test header(acq)[:protocol] == "t1_se"

    given = Header(; fov = (160, 120))
    acq = AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = given)
    @test header(acq) === given

    # A mismatch between fov, spacing and the image size is warned about, not rejected.
    @test_logs (:warn, r"fov") AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (1, 2), spacing = (1, 1)))
    @test_logs (:warn, r"entries") AcquisitionInfo(rand(ComplexF32, 16, 12); is3D = false, header = (; fov = (1, 2, 3)))

    settag!(acq, :subject, "s1")
    acq2 = AcquisitionInfo(acq; sensitivity_maps = nothing)
    @test header(acq2) === header(acq)
    @test gettag(acq2, :subject) == "s1"

    acq3 = AcquisitionInfo(acq; header = (; TR = 10))
    @test header(acq3).TR == 10
    @test isnothing(header(acq3).fov)
end

@testitem "Tags" tags = [:acquisition] begin
    using Test
    using Ristretto

    acq = AcquisitionInfo(rand(ComplexF32, 8, 8); is3D = false)
    @test isempty(tags(acq))
    @test settag!(acq, :site, "A") === acq
    @test gettag(acq, :site) == "A"
    @test gettag(acq, "site") == "A"
    @test gettag(acq, :missing, 0) == 0
    @test_throws KeyError gettag(acq, :missing)
    @test tags(acq) == Dict("site" => "A")
end

@testitem "reconstruct returns a ReconImage with the acquisition's header" tags = [:reconstruction] begin
    using Test
    using Ristretto
    using Ristretto: header, _spatial_ndims
    using NamedDims

    ksp = NamedDimsArray{(:kx, :ky, :z)}(rand(ComplexF32, 16, 16, 3))
    acq = AcquisitionInfo(ksp; is3D = false, header = (; fov = (160, 160), slice_spacing = 5))
    settag!(acq, :subject, "s1")
    img = reconstruct(acq)
    @test img isa ReconImage
    @test parent(img) isa NamedDimsArray
    @test dimnames(img) == (:x, :y, :z)
    @test _spatial_ndims(img) == 2
    @test header(img).spacing == (10.0, 10.0)
    @test isnothing(header(acq).spacing)   # derived on the image's copy only
    @test gettag(img, :subject) == "s1"

    # The image's header is a copy: tagging it leaves the acquisition alone.
    settag!(img, :subject, "other")
    @test gettag(acq, :subject) == "s1"

    @test Array(img) isa Array{ComplexF32, 3}
    @test unname(img) isa Array{ComplexF32, 3}
    @test img .* 2 ≈ 2 .* Array(img)
end

@testitem "ReconImage: keyword indexing moves the offset" tags = [:reconstruction] begin
    using Test
    using Ristretto
    using Ristretto: header, _spatial_ndims
    using NamedDims

    R = [0.0 0 1; 1 0 0; 0 1 0]          # x → L-P-S column 1 = (0,1,0), ...
    h = Header(; fov = (16, 12, 8), spacing = (2, 2, 2), orientation = R, offset = (10, 20, 30))
    img = ReconImage(NamedDimsArray{(:x, :y, :z, :time)}(rand(8, 6, 4, 3)), h)
    @test size(img, :time) == 3
    @test axes(img, :y) == Base.OneTo(6)

    crop = img[x = 3:6]
    @test crop isa ReconImage
    @test size(crop) == (4, 6, 4, 3)
    @test _spatial_ndims(crop) == 3
    @test header(crop).fov == (8.0, 12.0, 8.0)
    @test collect(header(crop).offset) ≈ [10, 20, 30] .+ R[:, 1] .* (2 * 2.0)

    slice = img[z = 3]
    @test _spatial_ndims(slice) == 2
    @test header(slice).spacing == (2.0, 2.0)
    @test header(slice).slice_thickness == 2.0
    @test collect(header(slice).offset) ≈ [10, 20, 30] .+ R[:, 3] .* (2 * 2.0)

    # Dropping x keeps the remaining in-plane axes first in the orientation.
    sag = img[x = 2]
    @test header(sag).orientation[:, 1] == R[:, 2]
    @test header(sag).orientation[:, 3] == R[:, 1]

    # Non-spatial axes leave the geometry alone; views work the same way.
    frame = view(img; time = 2)
    @test frame isa ReconImage
    @test header(frame).offset == h.offset
    @test header(img).offset == (10.0, 20.0, 30.0)   # slicing never changes the original
    @test _spatial_ndims(frame) == 3

    # A 2D multi-slice image: `z` moves the offset by the slice spacing.
    h2 = Header(; spacing = (1, 1), slice_spacing = 5, orientation = R, offset = (0, 0, 0))
    ms = ReconImage(NamedDimsArray{(:x, :y, :z)}(rand(8, 6, 4)), h2)
    @test _spatial_ndims(ms) == 2
    @test collect(header(ms[z = 4]).offset) ≈ R[:, 3] .* 15
    @test _spatial_ndims(ms[z = 4]) == 2
end

@testitem "ReconImage moves to a device with its header" tags = [:gpu, :reconstruction] setup = [GpuEnvSetup, GpuHelpers] begin
    using Test
    using Ristretto
    using Ristretto: header
    using NamedDims

    acq = AcquisitionInfo(NamedDimsArray{(:kx, :ky)}(rand(ComplexF32, 16, 16)); is3D = false, header = (; fov = (160, 160)))
    test_on_devices(acq; rtol = 1.0e-4) do a
        img = reconstruct(a)
        @test img isa ReconImage
        @test header(img).fov == (160.0, 160.0)
        img
    end
end

@testitem "ReconImage: array interface and spatial axes" tags = [:reconstruction] begin
    using Test
    using Ristretto
    using Ristretto: header, _spatial_ndims
    using NamedDims

    data = NamedDimsArray{(:x, :y, :time)}(rand(ComplexF32, 4, 3, 2))
    img = ReconImage(data, (; spacing = (2, 2)))
    @test _spatial_ndims(img) == 2
    @test NamedDimsArray(img) === data
    @test dimnames(img, 3) === :time
    @test convert(Array, img) == Array(data)
    @test similar(img) isa Array{ComplexF32, 3}
    img[1, 1, 1] = 5
    @test data[1, 1, 1] == 5
    c = copy(img)
    c[1, 1, 1] = 0
    @test img[1, 1, 1] == 5
    @test header(c) !== header(img) && header(c).spacing == (2.0, 2.0)
    @test occursin("spacing 2.0×2.0 mm", sprint(show, MIME"text/plain"(), img))
    @test occursin("(:x, :y, :time)", sprint(show, MIME"text/plain"(), img))

    # Without a header, the spatial axes are guessed from the fov or the dimension names.
    @test _spatial_ndims(ReconImage(rand(4, 3, 2), (; fov = (1, 2, 3)))) == 3
    @test _spatial_ndims(ReconImage(rand(4, 3, 2))) == 2
    @test _spatial_ndims(ReconImage(NamedDimsArray{(:x, :y, :z, :t)}(rand(2, 2, 2, 2)))) == 3
    @test_throws ArgumentError ReconImage(rand(2, 2); spatial_ndims = 3)
    @test_throws ArgumentError NamedDimsArray(ReconImage(rand(2, 2)))
    @test_throws ArgumentError ReconImage(rand(2, 2))[x = 1]

    # A selection that is not a range drops the offset; a single row leaves a profile, which
    # the header's geometry cannot describe.
    h = Header(; fov = (8, 6), spacing = (2, 2), orientation = [1.0 0 0; 0 1 0; 0 0 1], offset = (0, 0, 0))
    img2 = ReconImage(NamedDimsArray{(:x, :y)}(rand(4, 3)), h)
    picked = img2[x = [1, 3]]
    @test isnothing(header(picked).offset)
    profile = img2[y = 2]
    @test _spatial_ndims(profile) == 1
    @test isnothing(header(profile).fov) && isnothing(header(profile).spacing)
    @test img2[x = 1, y = 1] isa Real
end

@testitem "Header: dictionary interface" tags = [:acquisition] begin
    using Test
    using Ristretto
    using Ristretto: header

    h = Header(; TE = [1, 2], site = "A", tags = Dict(:reader => "B"))
    @test h.TE == [1.0, 2.0]
    @test header(h) === h
    @test gettag(h, :reader) == "B"
    @test haskey(h, :tags)
    @test Dict(collect(h)) == Dict(:TE => [1.0, 2.0], :site => "A", :tags => Dict("reader" => "B"))
    @test get(() -> 0, h, :TR) == 0
    @test get(() -> 0, h, :site) == "A"
    delete!(h, :site)
    delete!(h, :tags)
    @test !haskey(h, :site) && !haskey(h, :tags)
    h.extra = Dict("protocol" => "x")
    @test h[:protocol] == "x"
    h.fov = nothing
    @test !haskey(h, :fov)
    @test sprint(show, Header()) == "Header()"
end

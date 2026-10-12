using TestItems

@testmodule ExportCase begin
    using Ristretto

    export h, img, voxel_lps

    # Three slices of a multi-slice 2D image over two frames, in a rotated orientation: x runs
    # along LPS y, y along LPS z, the slices along LPS x.
    const R = [0 0 1.0; 1 0 0; 0 1 0]
    const h = Header(;
        fov = (48, 32), spacing = (2, 2, 4), orientation = R, offset = (10, -20, 5), slice_thickness = 3,
        TE = 4.2, TR = 10, flip_angle = 15, field_strength = 3, protocol = "test",
    )
    settag!(h, "subject", "s01")
    const img = ReconImage(
        NamedDimsArray{(:x, :y, :z, :time)}(ComplexF32.(reshape(1:(24 * 16 * 3 * 2), 24, 16, 3, 2)) .* cis(0.3f0)),
        h; spatial_ndims = 2,
    )
    # The LPS position of the 1-based voxel (i, j, slice).
    voxel_lps(i, j, k) = collect(h.offset) .+ R * ([i - 1, j - 1, k - 1] .* [2.0, 2.0, 4.0])
end

@testitem "export: layout and geometry shared by the writers" tags = [:export, :extension] setup = [ExportCase] begin
    using Test
    using Ristretto
    using Ristretto: _export_geometry, _json
    using LinearAlgebra: I

    g = _export_geometry(img)
    @test size(g.vol) == (24, 16, 3, 2) && g.extra_names == [:time] && g.nd == 2
    @test g.spacing == (2.0, 2.0, 4.0) && g.orientation == h.orientation
    @test g.affine * [4, 5, 2, 1] ≈ [voxel_lps(5, 6, 3); 1]

    # A 2D image without a slice axis gets one of length one; a bare header still exports.
    g = @test_logs (:warn,) _export_geometry(ReconImage(NamedDimsArray{(:x, :y, :echo)}(rand(8, 6, 2)), Header(); spatial_ndims = 2))
    @test size(g.vol) == (8, 6, 1, 2) && g.extra_names == [:echo]
    @test g.affine[1:3, 1:3] == I(3)
    @test g.affine * [4, 3, 0, 1] ≈ [0, 0, 0, 1]

    @test_throws ArgumentError _export_geometry(ReconImage(rand(4); spatial_ndims = 1))
    @test _json(Dict("a" => Any[1, 2.5], "b\"" => nothing, "c" => "x\ny")) == "{\"a\": [1, 2.5], \"b\\\"\": null, \"c\": \"x\\u000ay\"}"
end

@testitem "export: NIfTI with a JSON sidecar" tags = [:export, :extension] setup = [ExportCase] begin
    using Test
    using Ristretto
    using NIfTI
    using FileIO
    import JSON

    dir = mktempdir()
    path = save(File{format"NIfTI"}(joinpath(dir, "img.nii.gz")), img)
    v = niread(path)
    @test size(v) == size(img)
    @test eltype(v) == ComplexF32
    @test v.raw == Array(img)
    @test v.header.sform_code == 1 && v.header.qform_code == 0
    # The affine maps voxel indices to RAS: LPS with the first two axes negated.
    lps_to_ras = [-1, -1, 1]
    @test NIfTI.getaffine(v.header) * [4, 5, 2, 1] ≈ [lps_to_ras .* voxel_lps(5, 6, 3); 1]
    @test collect(NIfTI.voxel_size(v.header)) ≈ [2, 2, 4]

    sidecar = JSON.parsefile(joinpath(dir, "img.json"))
    @test sidecar["EchoTime"] ≈ 0.0042
    @test sidecar["RepetitionTime"] ≈ 0.01
    @test sidecar["FlipAngle"] == 15
    @test sidecar["MagneticFieldStrength"] == 3
    @test sidecar["protocol"] == "test"
    @test sidecar["Tags"]["subject"] == "s01"
    @test save(joinpath(dir, "plain.nii"), img; sidecar = false) == joinpath(dir, "plain.nii")
    @test isfile(joinpath(dir, "plain.nii"))
    @test !isfile(joinpath(dir, "plain.json"))
end

@testitem "export: DICOM series" tags = [:export, :extension] setup = [ExportCase] begin
    using Test
    using Ristretto
    using DICOM
    using FileIO
    import JSON

    files = save(joinpath(mktempdir(), "series", "img.dcm"), img; series_description = "test")
    @test length(files) == 6
    @test basename(files[2]) == "img_00002.dcm"
    # A single image keeps the name it was given.
    single = joinpath(mktempdir(), "one.dcm")
    @test save(single, img[z = 1, time = 1]) == [single]
    @test isfile(single)
    # Slices vary fastest, then frames: file 5 is slice 2 of frame 2.
    d = dcm_parse(files[5])
    @test d[(0x0020, 0x0032)] ≈ voxel_lps(1, 1, 2) atol = 1.0e-6
    @test d[(0x0020, 0x0037)] ≈ [0, 1, 0, 0, 0, 1]
    @test d[(0x0028, 0x0030)] ≈ [2, 2]
    @test d[(0x0018, 0x0050)] ≈ 3
    @test d[(0x0018, 0x0081)] ≈ 4.2
    @test d[(0x0028, 0x0010)] == 16 && d[(0x0028, 0x0011)] == 24
    @test d[(0x0020, 0x0100)] == 2
    @test JSON.parse(d[(0x0020, 0x4000)])["subject"] == "s01"
    pixels = d[(0x0028, 0x1053)] .* permutedims(d[(0x7FE0, 0x0010)])
    @test pixels ≈ abs.(Array(img)[:, :, 2, 2]) rtol = 1.0e-4
end

@testitem "export: MRD images" tags = [:export, :extension] setup = [ExportCase] begin
    using Test
    using Ristretto
    using MRIFiles
    using FileIO: save
    using LinearAlgebra: norm

    path = save(ISMRMRDFile(joinpath(mktempdir(), "img.h5")), img)
    HDF5 = MRIFiles.HDF5
    HDF5.h5open(path) do f
        data = read(f["dataset/image_0/data"])
        @test size(data) == (24, 16, 1, 1, 6)
        @test ComplexF32.(getindex.(data, :real), getindex.(data, :imag)) ≈ reshape(Array(img), 24, 16, 1, 1, 6)
        heads = read(f["dataset/image_0/header"])
        @test length(heads) == 6
        h5 = heads[5]
        @test collect(h5.matrix_size) == [24, 16, 1]
        @test h5.data_type == 7 && h5.image_type == 5
        @test h5.slice == 1 && h5.phase == 1 && h5.image_index == 5
        # MRD positions an image by its centre voxel, (n ÷ 2) from the first one.
        @test collect(h5.position) ≈ voxel_lps(13, 9, 2) rtol = 1.0e-6
        @test collect(h5.read_dir) ≈ [0, 1, 0] && collect(h5.slice_dir) ≈ [1, 0, 0]
        attributes = read(f["dataset/image_0/attributes"])
        @test occursin("<name>subject</name><value>s01</value>", attributes[1])
        @test h5.attribute_string_len == ncodeunits(attributes[1])
        xml = only(read(f["dataset/xml"]))
        @test occursin("<TE>4.2</TE>", xml)
        @test occursin("<H1resonanceFrequency_Hz>127732434</H1resonanceFrequency_Hz>", xml)
    end
end

@testitem "export: reversed slice direction and scaled orientation" tags = [:export, :extension] begin
    using Test
    using Ristretto
    using Ristretto: _export_geometry
    using DICOM
    using FileIO: save

    # A reversed slice direction and an orientation whose columns are not unit vectors: the
    # affine keeps both, the writers get unit directions and positive spacings.
    h = Header(; spacing = (1, 1, 6), orientation = [2.0 0 0; 0 2 0; 0 0 -1], offset = (0, 0, 0))
    img = ReconImage(NamedDimsArray{(:x, :y, :z)}(rand(Float32, 4, 3, 2)), h; spatial_ndims = 2)
    g = _export_geometry(img)
    @test g.spacing == (2.0, 2.0, 6.0)
    @test g.orientation == [1.0 0 0; 0 1 0; 0 0 -1]
    @test g.affine * [1, 1, 1, 1] ≈ [2, 2, -6, 1]
    d = dcm_parse(first(save(joinpath(mktempdir(), "img.dcm"), img)))
    @test d[(0x0018, 0x0088)] ≈ 6 && d[(0x0018, 0x0050)] ≈ 6
    @test d[(0x0028, 0x0030)] ≈ [2, 2]
end

@testitem "export: DICOM sequence parameters per echo" tags = [:export, :extension] begin
    using Test
    using Ristretto
    using DICOM
    using FileIO: save

    h = Header(; fov = (8, 6), TE = [2.0, 4.0, 6.0], TR = 10, flip_angle = [10, 20])
    img = ReconImage(NamedDimsArray{(:x, :y, :echo)}(rand(Float32, 4, 3, 3)), h; spatial_ndims = 2)
    files = save(joinpath(mktempdir(), "img.dcm"), img)
    ds = dcm_parse.(files)
    @test [d[(0x0018, 0x0081)] for d in ds] ≈ [2, 4, 6]
    @test all(d -> d[(0x0018, 0x0080)] ≈ 10, ds)
    # Two flip angles match no axis of the image, so none is written.
    @test all(d -> !haskey(d.meta, (0x0018, 0x1314)), ds)
end

@testitem "export: MRD into an existing file" tags = [:export, :extension] setup = [ExportCase] begin
    using Test
    using Ristretto
    using MRIFiles
    using FileIO: save

    HDF5 = MRIFiles.HDF5
    path = joinpath(mktempdir(), "scan.h5")
    HDF5.h5open(path, "w") do f
        write(f, "/dataset/xml", ["<ismrmrdHeader>raw</ismrmrdHeader>"])
        write(f, "/dataset/data", collect(1:4))
    end
    save(ISMRMRDFile(path), img)
    save(ISMRMRDFile(path), img[time = 1]; group = "image_1")
    save(ISMRMRDFile(path), img[time = 2]; group = "image_1")
    HDF5.h5open(path) do f
        @test read(f["dataset/data"]) == 1:4
        @test only(read(f["dataset/xml"])) == "<ismrmrdHeader>raw</ismrmrdHeader>"
        @test length(read(f["dataset/image_0/header"])) == 6
        @test length(read(f["dataset/image_1/header"])) == 3
    end
end

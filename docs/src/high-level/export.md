# Export

A [`ReconImage`](@ref) carries its geometry and metadata in its [`Header`](@ref), so writing it
to a file format needs nothing else. Saving follows the
[FileIO](https://github.com/JuliaIO/FileIO.jl) convention, `save(file, img)`; each format is a
package extension, loaded with its package:

| Call | Format | Load |
|---|---|---|
| `save("recon.nii", img)`, `save(File{format"NIfTI"}("recon.nii.gz"), img)` | NIfTI-1 and a BIDS-style JSON sidecar | `using FileIO, NIfTI` |
| `save("recon.dcm", img)` | DICOM MR image series | `using FileIO, DICOM` |
| `save(ISMRMRDFile("recon.h5"), img)` | MRD (ISMRMRD HDF5) images | `using MRIFiles` |

```julia
using Ristretto, FileIO, NIfTI

img = reconstruct(acq, IterativeReconstruction(TotalVariation2D(1e-3)))
settag!(img, :reader, "A")
save("recon.nii", img)                                   # also writes recon.json
save(File{format"NIfTI"}("recon.nii.gz"), img; sidecar = false)
```

FileIO picks the format from the extension; `.nii.gz` ends in `.gz`, which FileIO takes for
gzip, so a compressed NIfTI file names its format with `File{format"NIfTI"}`. An `.h5` file is
plain HDF5 to FileIO, so MRD output goes through MRIFiles' `ISMRMRDFile`, the way MRIFiles saves
raw data.

**NIfTI** (`sidecar = true`): the first three axes of the file are x, y and z; a 2D image gets a z
axis of length one, or its slices when the axis after `x` and `y` is named `:z` or `:slice`.
Further axes (time, echoes, ...) follow. Complex images are written as `complex64` or
`complex128`; save `abs.(img)` for a magnitude image. The sidecar next to the file holds the
sequence parameters under their BIDS names (`EchoTime`, `RepetitionTime`, ... in seconds), the
other header entries and the tags. Returns the path.

**DICOM** (`series_description = "Ristretto"`, `series_number = 1`): one file per slice and per
index of any non-image axis, `recon_00001.dcm`, `recon_00002.dcm`, ... next to the path given (a
single image keeps the name given), and returns the file names. The pixel data is the magnitude
as 16-bit integers with a rescale slope, so `RescaleSlope * stored` recovers it to 1 part in 65535
of the maximum. Patient and study fields are left empty except for generated UIDs; fill them in
with a DICOM tool before archiving.

**MRD** (`group = "image_0"`): images under `/dataset/<group>` with an `ismrmrdHeader` XML holding
the field of view, the matrix size and the sequence parameters. A 3D image is one MRD image, a 2D
image one per slice; each non-image axis index is another image, numbered by `image_index`, with
a `:time` index also in the `phase` counter. Returns the path.

## Geometry

The header stores positions in the patient coordinate system LPS, as MRD and DICOM do (see
[Metadata header](@ref)). DICOM and MRD files receive it unchanged; the NIfTI affine maps voxel
indices to RAS, negating the first two coordinates.

What the header lacks is filled in so that a file can always be written: an identity
orientation, 1 mm voxels (with a warning), a slice spacing equal to the slice thickness, and the
image centre at the scanner origin. An acquisition [read from MRD](acquisition_info.md#Building-from-raw-ISMRMRD-data-(MRIBase.RawAcquisitionData))
has all of it.

## What goes where

| Header | NIfTI | DICOM | MRD |
|---|---|---|---|
| `spacing`, `orientation`, `offset` | `sform` affine, `pixdim` | `PixelSpacing`, `ImageOrientationPatient`, `ImagePositionPatient` | `field_of_view`, `read_dir`/`phase_dir`/`slice_dir`, `position` |
| `slice_spacing`, `slice_thickness` | third axis of the affine | `SpacingBetweenSlices`, `SliceThickness` | slice `position` |
| `TE`, `TR`, `TI`, `flip_angle`, `field_strength` | sidecar, BIDS names, in seconds | `EchoTime`, `RepetitionTime`, ... | `ismrmrdHeader` XML |
| other keys | sidecar | — | — |
| tags | sidecar, `Tags` | `ImageComments` as JSON | meta attributes of each image |

NIfTI and MRD keep complex images complex; DICOM stores the magnitude.

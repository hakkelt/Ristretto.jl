module RistrettoMRIFilesExt

using Ristretto
using Ristretto: CartesianAcquisitionInfo, NonCartesianAcquisitionInfo, Header, ReconImage, header,
    _spatial_ndims, _export_volume, _lps_affine
using LinearAlgebra: dot, norm
using ArgCheck: @argcheck
using NamedDims: NamedDimsArray
using MRIFiles: MRIFiles, ISMRMRDFile
const MRIBase = MRIFiles.MRIBase
using .MRIBase: RawAcquisitionData, Limit, kspaceNodes
const FileIO = MRIFiles.FileIO
const HDF5 = MRIFiles.HDF5
const API = HDF5.API

# Gyromagnetic ratio of ¹H in Hz/T.
const _GAMMA_HZ_PER_T = 42.577478e6

include("raw_data.jl")
include("mrd_export.jl")

end # module RistrettoMRIFilesExt

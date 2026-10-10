using TestItemRunner
using Ristretto

# Run all tests. Example filtered runs from the package root:
#
#   julia --project=test -e '
#       using TestItemRunner
#       TestItemRunner.run_tests(pwd(); filter = ti -> :encoding in ti.tags)
#   '
#
# Available tags: :encoding, :regularization, :minimizer,
#                 :reconstruction, :integration, :nfft,
#                 :quality, :aqua, :jet, :acquisition, :simulation,
#                 :components, :operators, :preprocessing,
#                 :acquisition_info, :analysis, :fourier, :sensitivity_maps, :gpu,
#                 :export, :extension
#
# RISTRETTO_TEST_TAGS (comma-separated) runs only the items carrying one of the tags; CI uses it
# to collect the coverage of ext/ in a run of its own (`:extension`).

const TAGS = Symbol.(filter(!isempty, split(get(ENV, "RISTRETTO_TEST_TAGS", ""), ',')))

TestItemRunner.run_tests(pkgdir(Ristretto); filter = ti -> isempty(TAGS) || any(in(ti.tags), TAGS))

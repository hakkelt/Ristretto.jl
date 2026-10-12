# Download the datasets the real-data tutorials load, retrying a stalled transfer, so that the
# documentation build starts from a filled MRITestData cache. Run with the docs environment:
#
#   julia --project=docs docs/fetch_data.jl
using MRITestData

const DATASETS = (
    (MRITestData.M4RAW, "multicoil_train/2022062402_T203"),  # tutorial 09
    (MRITestData.OCMR_SOURCE, "fs_0001_1_5T"),  # tutorial 10
    (MRITestData.USC_SPEECH, "sub054/2drt/09_northwind1_r1"),  # tutorial 10
)
const ATTEMPTS = 5

MRITestData.get_download_path() === nothing && MRITestData.set_download_path!(:cache)

for (source, id) in DATASETS
    entry = MRITestData.dataset(source, id)
    for attempt in 1:ATTEMPTS
        try
            MRITestData.load_raw(entry)
            break
        catch err
            attempt == ATTEMPTS && rethrow()
            @warn "Downloading $id failed (attempt $attempt of $ATTEMPTS), retrying" exception = err
            sleep(10 * attempt)
        end
    end
end

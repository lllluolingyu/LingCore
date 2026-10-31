# Reviewed release metadata

`terminal-bench-4.0.0.json` contains public metadata only: no instructions,
solutions, verifier code, or task data. Resolve and validate read it offline.

The [publisher's v4.0.0 release](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0)
names commit `452bf305c6daa62fc59061d22133a7cbc7c1572e` and links the
[published Hub dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4).
The Hub metadata was read with Harbor 0.25.0's `PackageDatasetClient` on
2026-10-30. Its 66 task names match the release's Git tree. The catalog records
the immutable dataset version and each package content digest. Harbor's
`Packager.compute_content_hash` recomputes those digests from local exports.
The bridge also requires every local file to belong to that package file set:
ignored extra files could otherwise enter a Docker build without changing the
publisher's digest.

This proves correspondence to the publisher's Hub packages associated with the
release. It does not claim that Git task directories and Hub exports are byte
identical: Hub exports can use prebuilt images where Git uses Dockerfiles.
The manifest digest, release linkage, and verified membership enter the lock.
Changed packages and false release-source claims fail preflight. A partial
export can have `verified_subset` provenance, but cannot qualify for benchmark
comparison. Unknown releases remain usable for development as `unverified`.

The [official run guide](https://www.tbench.ai/run) declares five attempts per
task. This release has no dataset-level metric files; Harbor's default mean
aggregation and the release's binary `reward` contract are recorded in the
reviewed policy. LingCore additionally requires full membership, unchanged
task budgets, pinned images, required traces, and prohibited training use.
Eligibility under that local policy does not certify leaderboard acceptance.

To review an update, use the isolated pinned Harbor environment to call
`await PackageDatasetClient().get_dataset_metadata("terminal-bench/terminal-bench@4.0.0")`.
Compare the release tag, commit, linked Hub version, Git membership, package
digests, dataset metric files, and upstream run rules before updating this
catalog. Do not update it automatically from resolve/run, or let a suite supply
its own trusted manifest. A changed manifest is lock drift.

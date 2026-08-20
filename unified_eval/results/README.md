# Results

Each immediate subdirectory is a configuration-fingerprinted experiment run. A complete run may
contain:

- `run_config.json`: semantic and runtime configuration;
- `validation.json`: resolved input provenance;
- one subdirectory per evaluated task;
- `summary.json`: aggregate metrics;
- `checksums.json`: artifact hashes.

`grad` runs evaluate the original Grad approach developed in this project; they are not
reproduction runs. IA3 post-training and SN-Tune runs are the reproduced prior-work comparisons.

New run names should use `<task>_<method>_<key-settings>_<dtype>` so related sweeps sort together.
Benchmark-only runs should include `benchmark` in the name. Historical results that no longer
belong to the active experiment scope should be removed instead of mixed into current summaries.

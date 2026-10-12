# Detector input converter (2026-10-12)

`build_evenet_input_from_parquet.py` fits/applies a missingness-aware feature contract for the additive `particle_detector` version1 Parquet payload. It replaces the absent documented converter entrance for this schema; legacy inputs are unsupported. This is a usable feature-shard converter and bounded EveNet reader/encoder adapter, **not a complete training pipeline**: old classification, charge-ordered decay and invisible task targets are not inferred from modern truth codes. Original truth stays in source columns, outside model inputs. Task-specific target integration and an approved train/validation/test split remain prerequisites for training.

All input Arrow columns, their order, physical types and metadata are retained as a prefix. Tensor columns are appended. Event identity, original signed weight (also copied to event_weight/central_weight), truth and the complete raw detector JSON remain available. No luminosity normalization is repeated. Region selection uses saved `regions` metadata and `region_mask`, or explicit `all`; no nprong or PDG selection is added. Particles use ascending original MAIN index, with all arrays aligned and indices saved; a trackless or zero-pT real particle remains present. Capacity overflow fails rather than truncating. Production capacity is an explicit caller decision, not established by the smoke sample.

## Feature and validity contract

Selected values are MAIN energy, derived pT/eta/phi, charge, mass; named ELID tag/conversion, MUID word/chi2/hit pattern, RPROCO pion/kaon/proton/heavy/electron/quality, all named MTPC and HAID fields listed in `detector_inputs.PID`; singleton-track chi2/ndf and track count; MAIN TrackLength/DetectorMask; and separate EMNC, HCNC, HCAL and SSTC shower counts, energy sums and mean Cartesian positions. Source units are preserved. Centroids are unweighted means over distinct linked shower indices, not fitted particle directions. These compact summaries avoid a variable-width shower encoder; HCNC and HCAL are never combined.

Detailed profiles, covariance/helix states, TOF, muon chamber arrays, PHOT/pi0/conversion graph details and other raw payload fields remain **raw-only** in the retained JSON. They are not claimed to be model features. MAIN.PDG and all truth-derived species are excluded from x.

Each selected value has stored_valid, measurement_known, measurement_valid and normalization_supported channels. A stored finite value with unknown semantic validity is available as raw information with explicit unknown flags; it is never certified as a measurement. Known-false values are excluded from fitted statistics and model value channels, but preserved in raw_detector_values. Derived values propagate false if any source is false, true only if every source is true, otherwise unknown. Missing/nonfinite values are unavailable; valid zero is retained. PID/track/calo association onehots distinguish linked, no_association, unavailable, malformed, ambiguous and known empty. Malformed or unavailable shower ancestry suppresses incomplete sums. Multiple singleton PID/track matches do not produce a guessed scalar.

Normalization is unweighted, feature-wise pooled centered moments over usable **training-only** values, including zeros. Charge remains raw. Features absent in training have support=false and zero model value even if present later. Frozen means/stds/counts, feature-order fingerprint, selection/capacity contract, source schema hashes and training identity digest are recorded. Fit rejects data; `--role train` and an owner-provided `--split-id` are mandatory. The converter does not create or independently certify disjoint splits: pass already approved training files only, then apply the same artifact to validation/test/data. Every generated backend normalization tag is `none`; do not run a second automatic normalization fit.

The current grouped encoder retains seven projected slots: four momentum, two learned detector slots and raw charge. Mask/status channels enter the detector group without normalization. Momentum missingness remains visible there; it never becomes particle padding. This confirms input/projection compatibility, not task-head or physics performance.

## Commands

Use an environment with the repository's dependencies and initialized submodules. Example (fresh outputs):

```sh
python build_evenet_input_from_parquet.py fit --input TRAIN.parquet --output TRAIN_features --normalization normalization.json --role train --split-id APPROVED_TRAIN --region all --max-particles CAPACITY
python build_evenet_input_from_parquet.py apply --input VALIDATION.parquet --output VALIDATION_features --normalization normalization.json --role validation --split-id APPROVED_VALIDATION --region all --max-particles CAPACITY
python smoke_detector_inputs.py TRAIN_features --rows 2
python test_detector_inputs.py
```

`load_tensor_batch` is the explicit adapter passed with `functools.partial(..., unflatten=unflatten_dict)` to current Core `process_event_batch`, as executed by the smoke command. It loads only named tensor columns and restores explicit zero-width conditions plus false conditions_mask. The default reader alone does not handle this empty-global convention; training integration must invoke this adapter and supply task targets. Generated analysis_detector.yaml contains the feature grouping fragment, not a complete training task configuration.

## Bounded evidence

`../artifacts/ml-detector-converter-20261012/` contains commands/logs/reports. The genuine fixture test skips explicitly on clean clones without the external fixture; set DETECTOR_INPUT_FIXTURE to enable it. All six actually ran in the local evidence. Six regressions exercise valid zeros, missing/unknown/false semantics and derived propagation, malformed/ambiguous/no-track associations, normalization support and stable moments, MAIN alignment/overflow/empty events, truth isolation, actual Core preprocessing and seven-slot projection. A genuine saved ten-MC-row Parquet regression checks all **513** source column values/types/metadata survive write/read.

The fresh merged fixture separately captures two tau and two data rows (MAIN counts 14,4,3,3) from the new NERSC production. Its twelve selected source columns/identity/weights are checked exactly after conversion; JSON capture does not preserve physical Arrow types, so it is not the 513-column type check. Capacity16 avoids truncation. Normalization fitted to two tau fixture rows is explicitly engineering-only, not a production training artifact. No full conversion, training or new Slurm allocation was run.

# Provenance

The confirmatory and external studies were run against frozen protocol/code hash manifests stored under `configs/`.

The repository preserves negative results and runtime deviations rather than rewriting the record after outcomes were known.

One limitation applies to `results/derived/pusht_mechanism_posthoc.json`: the original generated JSON bytes were not retained after the Colab run. The file here was reconstructed exactly from the terminal-emitted JSON. It is therefore useful as a derived diagnostic record, but it is not represented as the original generated artifact.

The PushT post-hoc mechanism analysis does not alter the preregistered external verdict.

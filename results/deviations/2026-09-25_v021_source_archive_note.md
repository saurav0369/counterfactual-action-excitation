# v0.2.1 source archive note

The current repository preserves the v0.2.1 protocol, manifest, frozen hash list, raw outputs, and derived analysis. However, the present `src/model.py` no longer matches the hash recorded in `configs/confirmatory_freeze.sha256` for the v0.2.1 run. File timestamps show that `src/model.py` was modified after the v0.2.1 raw confirmatory output was written and before the v0.3 freeze was created.

Therefore, the current tree cannot reproduce the exact v0.2.1 executable source byte-for-byte. This is an archival weakness and should be disclosed if v0.2.1 is discussed. It does not affect the v0.3 positive result or the PushT external result: all files covered by `configs/v03_confirmatory_freeze.sha256` and `configs/pusht_external_freeze.sha256` match the current repository exactly.

v0.2.1 is retained as a failed stricter test and is not the basis of the main positive claim.

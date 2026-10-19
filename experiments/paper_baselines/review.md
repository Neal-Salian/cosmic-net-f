# Independent review and execution record

A read-only independent review checked the experiment code and first complete
run against the user mission. It independently verified all input/output
hashes, 12,294 prediction/residual records, 9,670 exported masks, matched
populations and budgets, and physical-node coverage of all test masks. It
confirmed that averaging squared errors within halos before a paired halo
bootstrap avoids pseudoreplication and found no other important numerical
or statistical issue.

Two important findings were fixed with failing-then-passing regressions:

1. Alternative gradient forwards used a private meter and were omitted from
   the shared counter. They now use the shared meter; the final run records
   164 scoring forwards (82 policy-score replays plus 82 gradient forwards).
2. The recovered policy-training checkpoint was hashed but its expected
   checksum was not compared with the available stage B provenance entry.
   The final run checks that checksum before parsing/using its split IDs.

The full frozen-model experiment was rerun into `results/run-20261016-final-v2`
after both fixes. The earlier executed run remains at
`results/run-20261016-final` for provenance; it is superseded for call-count
and recovered-metadata audit claims. Its predictions/masks remain exported.
The two initial failed preflight manifests remain separately labeled failed.
No model was retrained and no favorable seed or halo was selected.

Dependency versions now explicitly include PyG, SciPy, NetworkX, pandas,
matplotlib, PyYAML, python-dotenv, requests and pytest, in addition to Python,
PyTorch and NumPy. Numerical source fingerprints and result hashes are in
the final run and analysis manifests.

The reviewer explicitly declined to certify original predictor leakage and
target semantics, independent policy-training variance, GPU/deployment
speedups, or external/large-graph generalization. These are unavailable
scientific claims, not hidden positive findings; the evidence documents
retain each limitation. No second review was substituted for regression
verification of the identified fixes.

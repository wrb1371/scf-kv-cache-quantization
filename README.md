# SCF anonymous review artifact

This package contains the implementation snapshot, exact paper protocol, audit
utilities, and the source data/scripts for all SCF figures. Model weights,
datasets, fitted codebooks, and predictions are excluded.

## Environment

Use Python 3.11 with PyTorch, Transformers, Datasets, NumPy, SciPy, and
Matplotlib. The experiments use local copies of the model checkpoints and the
standard Hugging Face C4 and WikiText-2 datasets.

## Main entry points

- `src/calibrate_gsrq_codebooks.py`: residual-VQ calibration with SCF.
- `src/calibrate_m20_cq.py`: grouped-VQ paired calibration (the `m20` filename
  is retained only for compatibility with sealed artifacts).
- `src/calibrate_m20_fibquant.py`: shared-block FibQuant adapter.
  Pass `--norm-squared-weighting` to reproduce the strict raw-coordinate
  transport boundary; the reported method uses normalized-direction geometry.
- `src/run_gsrq_ppl.py` and `src/run_lowrate_vq_ppl.py`: evaluation.
- `src/summarize_lowrate_vq_pair.py`: paired robustness statistics.
- `configs/paper_protocol.json`: self-contained paper settings.

Representative hashed calibration-window manifests are in
`configs/calibration_manifests/`, and the frozen norm-squared attribution is in
`results/`. Full commands follow the same arguments in
`configs/paper_protocol.json`. Machine-local paths are forbidden in this
generated tree; run `python prepare_anonymous_scf_release.py` from the parent
project to rebuild and audit it.

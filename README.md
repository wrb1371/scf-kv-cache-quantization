# SCF anonymous review artifact

This package contains the SCF implementation, the exact paper protocol,
LongBench and efficiency runners, focused tests, and source data/scripts for
the paper figures. Model weights, datasets, fitted codebooks, and predictions
are excluded.

## Reference environment

The runtime measurements used Python 3.10.20, PyTorch 2.5.1 with CUDA 12.1,
and Transformers 4.45.1 on an NVIDIA A100 80GB PCIe GPU. Install the Python
dependencies with `pip install -r requirements.txt`. Experiments use local
model checkpoints and the standard C4, WikiText-2, and LongBench datasets.

## Main entry points

- `src/calibrate_gsrq_codebooks.py`: residual-VQ calibration with SCF.
- `src/calibrate_m20_cq.py`: grouped-VQ paired calibration (the `m20` filename
  is retained only for compatibility with sealed artifacts).
- `src/calibrate_m20_fibquant.py`: shared-block FibQuant adapter.
  Pass `--norm-squared-weighting` to reproduce the strict raw-coordinate
  transport boundary; the reported method uses normalized-direction geometry.
- `src/run_gsrq_ppl.py` and `src/run_lowrate_vq_ppl.py`: evaluation.
- `src/run_gsrq_longbench.py`: GSRQ/SCF-GSRQ packed-cache LongBench.
- `src/run_m20_backend_longbench.py`: CQ/SCF-CQ and FibQuant/SCF-FibQuant
  packed-cache LongBench. The `m20` method aliases are retained for artifact
  compatibility.
- `src/benchmark_scf_efficiency.py`: clean-process end-to-end latency and
  memory benchmark for FP16, GSRQ, CQ, and FibQuant.
- `src/benchmark_fibquant_paired_efficiency.py`: interleaved FibQuant pair.
- `src/benchmark_scf_encoder_components.py`: encoder microbenchmarks.
- `src/summarize_lowrate_vq_pair.py`: paired robustness statistics.
- `configs/paper_protocol.json`: self-contained paper settings.

Representative hashed calibration-window manifests are in
`configs/calibration_manifests/`, and the frozen norm-squared attribution is in
`results/`. Accepted raw runtime repetitions and the optimized FibQuant summary
are in `results/runtime/`; all machine-local paths have been redacted.

## Quick checks

From the repository root:

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q src/test_lowrate_vq_codecs.py tests
python verify_release.py
```

All runners expose their complete command-line interface through `--help`.
Pass explicit paths for the model checkpoint, frozen codec artifacts, dataset,
official LongBench checkout, and output directory. For example, the grouped-VQ
runner uses `--method cq`, `m20_cq`, `fibquant`, or `m20_fibquant`; the GSRQ
runner uses `--method gsrq` or `gsrq_scf`.

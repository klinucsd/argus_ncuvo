# Test results

## CDeep3M layer on NRP pods

Tested with the same CDeep3M layer on the plain Jupyter base (`kaiucsd/cdeep3m-nrp:0.1-cpu` and
`0.1-cu118`), CDeep3M-anywhere at commit `475277e`, using `pod_test.sh`.

| | CPU pod (4 threads), 2026-11-19 | GPU pod (RTX 2080 Ti, 11 GB), 2026-11-20 |
|---|---|---|
| `pod_test.sh` | ALL STEPS PASSED | ALL STEPS PASSED (`DEVICE=vulkan`); `gpu_check.sh` ALL OK |
| Prediction, 3 images | 22 s (int8), Dice 0.7142 (= CDeep3M-anywhere's own result) | 18 s (Vulkan, fp32), Dice 0.7247 |
| Training, per update (`--iter-size 1`) | ~11 s | 0.2 s; same losses as on the CPU |
| 50-update fine-tune (`--iter-size 2`, lr 0.001) | — | 0.4 s/update, peak 2.9 GB GPU memory, loss 0.16 → 0.12 |

Notes:
- CPU training is slow partly because the trainer's `--threads` defaults to 8; pass the pod's CPU count.
- At 0.4 s/update a 3,000-update fine-tune takes about 20 minutes on an RTX 2080 Ti. 1fm training fits
  easily in 11 GB; 3fm and 5fm are untested.
- On NRP, getting a GPU is the constraint, not its size: request "Any" GPU with small CPU and memory.

## ARGUS + CDeep3M (`kaiucsd/argus-ncuvo:0.1-cpu`, ARGUS `v1.5.11-rc3`), built 2026-11-20

Checked on the build machine (no GPU):
- Python packages: the CDeep3M layer adds 14 (PyTorch, ncnn, OpenCV, scikit-image and their
  dependencies) and changes none of ARGUS's 350. `pip check` reports only two OpenTelemetry
  version warnings that are already present in plain ARGUS.
- ARGUS: `%%ask` and `%%skill` magics load, deepagents 0.7.24, 18 core skills, JupyterLab 4.2.4.
- CDeep3M: tools on `PATH` (including `gpu_check.sh`), trainer imports.

Pod test of this image: pending.

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

Published:

| Tag | Size | Digest |
|---|---|---|
| `kaiucsd/argus-ncuvo:0.1-cpu` | 5.12 GB | `sha256:02de81130cb4b66ca9579859ec32913c1a7184f0bfb82fb7a5e3054bcf4f081f` |
| `kaiucsd/argus-ncuvo:0.1-cu118` | 9.81 GB | `sha256:eb4fcf30cf9da2f051568181b4560be7928681d0823945070be188ab203b4467` |

## Pod test of `kaiucsd/argus-ncuvo:0.1-cu118` (NRP, RTX 2080 Ti, 2 CPU threads), 2026-11-20

`notebooks/argus_ncuvo_pod_test.ipynb`, all four steps passed:
1. `gpu_check.sh`: ALL OK (CUDA and Vulkan).
2. ARGUS `%%ask`: identified the GPU and ran a test calculation on it with the image's CUDA PyTorch.
3. `DEVICE=vulkan pod_test.sh`: ALL STEPS PASSED; prediction 16 s on Vulkan, Dice 0.7247; training
   0.2 s/update with the same losses as before.
4. ARGUS on CDeep3M's output: per-image Dice 0.6791 / 0.7895 / 0.7032, overall 0.7247 (the test's
   own figure), side-by-side figure. Slips to fix in a skill: no colour key on the agreement map;
   the prediction column shows the probability map under a "p>0.5" heading.

## The cdeep3m skill on the build machine (CPU, 8 threads), 2026-11-20

Full cycle on the sample data in `kaiucsd/argus-ncuvo:0.1-cpu` with the skill mounted:
- `predict` with the pretrained SBEM mitochondria model (downloaded on first use, 13 s): 10 validation
  images in 25 s; `evaluate`: overall Dice 0.7241 (first three images identical to the GPU run).
- `prepare_training` (20 slices, 18 s, 2.7 GB); `train` plan, start in the background, a second start
  refused, `stop_training` at update 12 with progress saved, resumed from update 12 to 30 (3.1 s/update);
  `training_status` with loss curve; `finish_training` → model `mito-ft`; `predict` + `evaluate` with it:
  Dice 0.7075 (30 updates on 20 slices; CDeep3M-anywhere reports the same kind of small change).
- Fixed during testing: a finished training process was still reported as running (unreaped child).

Published with the skill:

| Tag | Size | Digest |
|---|---|---|
| `kaiucsd/argus-ncuvo:0.2-cpu` | 5.12 GB | `sha256:0f4f335d75fabd46a07f06594a4aa39f16635d66e161c9642f726d77090365e7` |
| `kaiucsd/argus-ncuvo:0.2-cu118` | 9.81 GB | `sha256:4ac6384a9d12d94c42f678210dac4c2e2e2531ee82c1cd9eb23e59032bfa12f9` |

Pod test of the skill through ARGUS (`%%ask`): pending.

# ARGUS-NCUVO

ARGUS — natural-language analysis in Jupyter notebooks — packaged for the **NCUVO** project
(National Connectomics and Ultrastructure Volume Observatory), with CDeep3M-anywhere for
segmenting electron-microscopy volumes built in. The aim is an agent that talks a researcher
through a segmentation workflow — which model, test on a few slices, fine-tune on labelled
slices, predict the volume — and runs it in the same NRP JupyterHub pod.

Like ARGUS-FEDER, this repository adds a deployment's own tools on top of a released ARGUS
image instead of changing ARGUS itself.

## The image

`kaiucsd/argus-ncuvo:<version>-cpu` and `-cu118`: an ARGUS release image plus

- [CDeep3M-anywhere](https://github.com/CRBS/CDeep3m-anywhere) (CRBS/NCMIR), pinned to a commit:
  CDeep3M2's pipeline with ncnn prediction (CPU or Vulkan GPU) and a PyTorch trainer
- PyTorch: CPU build, or CUDA 11.8 (NVIDIA GPUs from the GTX 10 series to the RTX 40 series)
- the ncnn conversion tools and the CDeep3M Caffe fork's `caffe.proto`

No model weights or data are in the image. Adding CDeep3M only adds Python packages; it changes
none of ARGUS's.

| Folder | What |
|---|---|
| `image/Dockerfile` | the build recipe; build commands in its header |
| `image/pod_test.sh` | CDeep3M-anywhere's acceptance test, adapted to run inside a pod (no Docker) |
| `docs/test_results.md` | results on NRP CPU and GPU pods |

## Quick check in a pod

```bash
gpu_check.sh                                  # GPU pods: CUDA (training) and Vulkan (prediction)
pod_test.sh /tmp/cdeep3m-test                 # CPU: convert, predict, train 3 updates, export
DEVICE=vulkan pod_test.sh /tmp/cdeep3m-test   # GPU pod: prediction on the GPU
```

A pass ends with `== ALL STEPS PASSED`. The test downloads the pretrained SBEM mitochondria model
(653 MB) and writes about 3 GB of training data into the folder given.

## Licenses

CDeep3M-anywhere and CDeep3M2 are under the UC Regents non-commercial license; see their
repositories.

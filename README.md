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
- the **`cdeep3m` skill** (`skills/cdeep3m/`), so the agent can do the work from plain-English requests
- the **`cil` skill** (`skills/cil/`): the Cell Image Library's public records, searchable, with downloads

No model weights or data are in the image. Adding CDeep3M only adds Python packages; it changes
none of ARGUS's.

| Folder | What |
|---|---|
| `skills/cdeep3m/` | the segmentation skill: `SKILL.md`, `cdeep3m_lib.py` (one function per task), and `cdeep3m_models.json`, the CDeep3M model zoo as a catalog |
| `skills/cil/` | the Cell Image Library skill: `SKILL.md`, `cil_lib.py`, and `cil_records.parquet`, a catalog of 10,023 public records (1.3 MB) |
| `tools/` | the scripts that build the two catalogs |
| `image/Dockerfile` | the build recipe; build from the repository root (commands in its header) |
| `image/pod_test.sh` | CDeep3M-anywhere's acceptance test, adapted to run inside a pod (no Docker) |
| `docs/test_results.md` | results on NRP CPU and GPU pods |

## The cdeep3m skill

The agent calls tested functions instead of writing its own segmentation code:

| Task | Function |
|---|---|
| hardware, folders, time estimates | `environment()` |
| models of the CDeep3M model zoo, by name or CIL model ID | `models(target=...)` |
| which model suits some data (pixel size, imaging) | `suggest_models(images, target=...)` |
| list or change settings | `settings()`, `set_defaults(...)` |
| models available | `models()` |
| segment images: a folder, or a Cell Image Library record | `predict(images, model=...)` |
| score against expert labels | `evaluate(prediction, labels)` |
| figure: image, probability, mask, label, agreement | `show(prediction, labels=...)` |
| training data from labelled slices | `prepare_training(images, labels)` |
| fine-tune or train (plan first, then in the background) | `train(model, data, start=...)` |
| follow / stop / continue training | `training_status()`, `stop_training()`, `train(..., start=True)` |
| make the trained network a model | `finish_training(run)` |

Every setting (networks, prediction speed, device, threshold, updates, learning rate, batch,
threads, …) can be given for one run as an argument, or for the whole project with
`set_defaults()`; every report starts with the settings it used. Models, predictions and training
runs are kept in a `cdeep3m/` folder next to the notebook.

## The cil skill

The [Cell Image Library](https://www.cellimagelibrary.org) serves its files without an account. The
skill ships a catalog of its public records, built from CIL's public metadata
([CIL_Public_Data_JSON](https://github.com/slash-segmentation/CIL_Public_Data_JSON), snapshot of
April 2018, so newer records are missing): imaging method, cellular component, organism, cell type,
pixel size and dimensions as recorded, every file with its size, terms of use and citation.

| Task | Function |
|---|---|
| what the library holds | `about()` |
| find records by subject, component, imaging, pixel size, size, terms | `search(...)` |
| one record: description, files, terms of use, citation | `record(id)` |
| what is inside a record's ZIP, read from the server without downloading | `contents(id)` |
| download a file, or only some files from inside its ZIP | `download(id, file=..., members=[...])` |
| a folder of greyscale 2-D slices, ready to segment | `slices(id)` |

Files larger than 2 GB are downloaded only after confirmation; interrupted downloads continue where
they stopped. The CIL server supports byte-range requests, which is how `contents()` and
`download(members=...)` read parts of large files.

The two catalogs are rebuilt with `tools/build_cil_records.py` and `tools/build_model_catalog.py`
(the latter lists each model archive's files from its tar headers, without downloading it).

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

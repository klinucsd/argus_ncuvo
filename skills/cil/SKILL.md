---
name: cil
description: Cell Image Library (CIL) — search 10,000 public microscopy records (electron, confocal, fluorescence, light; images, Z-stacks, videos) by subject, cellular component (mitochondria, synapses, membranes), organism, imaging method, pixel size, size or terms of use; read a record's details, files and citation; look inside its ZIP without downloading; download files or get image slices ready for segmentation. No account needed.
---

# Cell Image Library

Everything is done by the functions of `cil_lib`, shipped with this skill, with its catalog of
10,023 public records (CIL's public metadata snapshot of April 2018; newer records are not in it).
Each function prints a Markdown report: present it as printed, under its own heading (do not add
another heading above it), with at most one sentence of your own around it. Talk about records,
images and files in plain words; function names and code are for your scripts, not for the answer.

```python
import importlib, os, sys
try:
    import cil_lib as cil
except ImportError:                    # skill copied in without the image's search path
    sys.path.insert(0, os.path.expanduser("~/.deepagents/agent/skills/cil"))
    import cil_lib as cil
importlib.reload(cil)
```

This file and the table below are the whole interface: call the functions directly, without opening
`cil_lib.py` or the catalog file first.

**Do only what the request asks, in one script.** Searching, reading a record and listing what is inside
a ZIP download nothing. Files are downloaded only when the user asks for the data.

| The user wants to … | Call (exact argument names) |
|---|---|
| know what the library holds | `cil.about()` |
| find records | `cil.search(text=..., imaging=..., component=..., organism=..., cell_type=..., kind=..., z_stack=..., pixel_nm=..., max_gb=..., terms=..., limit=...)` — pass only what the user asked for |
| see one record: description, files, sizes, terms of use, citation | `cil.record(cil_id)` |
| see what is inside a record's ZIP | `cil.contents(cil_id)` |
| download a record's data | `cil.download(cil_id, file=...)`; its full original dataset (the extra file, when it has one): `cil.download(cil_id, file="extra")`; only some files from its ZIP: `cil.download(cil_id, members=[...])` |
| get a record's images as slices for segmentation | `cil.slices(cil_id)` |

A record ID can be written `7605`, `"CIL_7605"` or `"CIL:7605"`; functions also accept what an earlier
function returned. `pixel_nm` is a size in nm (within a factor of 1.5) or a `(low, high)` range.

## Rules

1. **Large downloads need the user's go-ahead.** Above 2 GB, `download` and `slices` show the size and
   download nothing. Tell the user the size and ask; download only after they agree, in a later request,
   with `confirm=True`.
2. **Mention the terms of use** when the user is going to use data: they differ per record (public domain,
   CC BY, non-commercial, copyright).
3. Pixel sizes, dimensions and descriptions are as the contributor recorded them, and many records have
   none. Say "as recorded" and do not fill gaps with guesses.
4. Downloads go to the temporary folder `/tmp/cil` (large; lost when the pod stops). An interrupted
   download continues where it stopped when it is run again.
5. To segment the images, the `cdeep3m` skill takes a CIL record ID directly.

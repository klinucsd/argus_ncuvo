"""CDeep3M segmentation of microscopy images, for agents.

One function per task. Each prints a short Markdown report that starts with the settings it used,
so an answer can show exactly how a result was produced, and returns a dict for further work.

    import cdeep3m_lib as cd
    cd.environment()                                   # hardware, folders, time estimates
    cd.settings()                                      # every setting, its value and meaning
    cd.set_defaults(updates=5000)                      # change a default for this project
    cd.models(target="mitochondria")                   # the CDeep3M model zoo and own models
    cd.suggest_models(11923, target="mitochondria")    # which model fits these data (CIL record or folder)
    cd.predict("images/", model="sbem-mitochondria")   # segment; model by name or CIL ID (50689)
    cd.predict(11923, model=50681)                     # images straight from a Cell Image Library record
    cd.prepare_training("train/images", "train/labels", name="mito")
    cd.train("sbem-mitochondria", "mito", name="mito-ft")             # shows the plan only
    cd.train("sbem-mitochondria", "mito", name="mito-ft", start=True) # starts in the background
    cd.training_status("mito-ft"); cd.stop_training("mito-ft")
    cd.finish_training("mito-ft")                      # -> a model usable by predict()
    cd.evaluate("<prediction name>", "labels/"); cd.show("<prediction name>", labels="labels/")
    cd.compare(["<prediction 1>", "<prediction 2>"], "labels/")   # side by side, per-image changes

Built on CDeep3M-anywhere (CRBS/NCMIR): CDeep3M2's pipeline, ncnn prediction, PyTorch trainer.
"""
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

import numpy as np

LIB_VERSION = 9
CDEEP3M2 = os.environ.get("CDEEP3M2", "/opt/cdeep3m2")
ANYWHERE = os.environ.get("CDEEP3M_ANYWHERE", "/opt/cdeep3m-anywhere")
FRAMES = ("1fm", "3fm", "5fm")
IMAGE_EXT = (".png", ".tif", ".tiff")

# Pretrained models: the CDeep3M model zoo (https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models), as a
# catalog shipped with this skill (cdeep3m_models.json, built by tools/build_model_catalog.py). Each
# model is known by a short name and by its CIL model ID.
def _load_zoo():
    try:
        doc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdeep3m_models.json")))
        return {m["name"]: m for m in doc["models"]}
    except (OSError, ValueError, KeyError):
        return {"sbem-mitochondria": {
            "id": 50681, "name": "sbem-mitochondria", "title": "SBEM and ssTEM Mitochondria", "target": "mitochondria",
            "about": "Mitochondria in serial block-face SEM and serial-section TEM images", "imaging": "SBEM, TEM",
            "pixel_nm": 4.8, "slice_nm": 70, "doi": "10.7295/W9CDEEP3M50681", "bytes": 684759040,
            "url": "https://cildata.crbs.ucsd.edu/media/model_display/50681/SBEM_Mitochondria.tar",
            "networks": ["1fm", "3fm", "5fm"], "test_entry": False}}


ZOO = _load_zoo()


def _model_key(x):
    """A model's name from its name, its CIL model ID (50689, "50689", "CIL_50689") or an earlier result."""
    if isinstance(x, dict):
        x = x.get("model") or x.get("name")
    if x is None:
        return None
    m = re.fullmatch(r"\s*(?:CIL[_: ]?|CDEEP3M)?(\d+)\s*", str(x), re.IGNORECASE)
    if m:
        for k, z in ZOO.items():
            if z.get("id") == int(m.group(1)):
                return k
        raise ValueError("no model with CIL ID %s in the model zoo; models() lists them" % m.group(1))
    return str(x)


# Every setting: default and meaning. Functions take any of them as keyword arguments;
# set_defaults() changes them for the whole project.
SETTINGS = {
    "frames": ("1fm", "Networks to use: 1fm (one slice), 3fm, 5fm (3 or 5 neighbouring slices), or a "
                      "combination such as 1fm,3fm,5fm, whose predictions are averaged (more accurate, slower)"),
    "augspeed": (1, "Prediction: how many flipped and rotated copies of each image are predicted and averaged. "
                    "1 = all (best quality), 2, 4, 10 = one copy (fastest)"),
    "device": ("auto", "auto (the GPU when there is one), gpu or cpu"),
    "precision": ("auto", "Prediction on the CPU: auto (int8 when the model has an int8 version), fp32 or int8"),
    "threshold": (0.5, "Probability above which a pixel counts as the object, for masks and scores: 0 to 1, "
                       "or on CDeep3M's 0-255 image scale (128 = 0.5)"),
    "updates": (3000, "Training: number of updates"),
    "learning_rate": (0.001, "Training: starting learning rate (0.001 for fine-tuning a model; "
                             "0.01 when training from scratch)"),
    "iter_size": (2, "Training: batches accumulated per update (memory stays the same; time grows with it)"),
    "batch": (5, "Training: image patches per batch"),
    "threads": (None, "CPU threads; empty = the CPUs this pod actually has"),
    "checkpoint_every": (100, "Training: save a checkpoint every this many updates (it also saves every 10 minutes)"),
}


# ----------------------------------------------------------------------------- folders and settings
def _sage_output():
    """ARGUS's output folder for this notebook (a variable in the notebook kernel), or None."""
    v = os.environ.get("SAGE_OUTPUT_DIR")
    if not v:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            v = ip.user_ns.get("SAGE_OUTPUT_DIR") if ip else None
        except Exception:
            v = None
    return v if v and os.path.isdir(v) else None


def _workspace():
    """The project folder. Under ARGUS it sits next to the notebook, not in ARGUS's output folder,
    whose files ARGUS clears when a cell is rerun: models and training runs must survive that."""
    w = os.environ.get("CDEEP3M_WORK")
    if not w:
        out = _sage_output()
        w = os.path.join(os.path.dirname(out.rstrip("/")) if out else os.getcwd(), "cdeep3m")
    os.makedirs(w, exist_ok=True)
    return w


def _for_display(path):
    """A copy of a figure in ARGUS's output folder, where answers can embed it (the original stays in
    the project folder)."""
    out = _sage_output()
    if not out:
        return path
    dest = os.path.join(out, os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path))
    shutil.copy(path, dest)
    return dest


def _scratch():
    s = os.environ.get("CDEEP3M_SCRATCH", "/tmp/cdeep3m")
    os.makedirs(s, exist_ok=True)
    return s


def _models_dir():
    """The project's own trained models (small, kept)."""
    m = os.environ.get("CDEEP3M_MODELS") or os.path.join(_workspace(), "models")
    os.makedirs(m, exist_ok=True)
    return m


def _zoo_dir():
    """Downloaded pretrained models: large and re-downloadable in seconds, so kept in the temporary
    folder rather than on the project's (often small) persistent storage."""
    z = os.path.join(_scratch(), "pretrained")
    os.makedirs(z, exist_ok=True)
    return z


def _model_path(name):
    return os.path.join(_zoo_dir() if name in ZOO else _models_dir(), name)


def _free_gb(path):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return float("nan")


def _settings_file():
    return os.path.join(_workspace(), "settings.json")


def _saved():
    try:
        return json.load(open(_settings_file()))
    except (OSError, ValueError):
        return {}


def _effective(overrides):
    bad = [k for k in overrides if k not in SETTINGS]
    if bad:
        raise ValueError("unknown setting(s) %s; known: %s" % (", ".join(bad), ", ".join(SETTINGS)))
    s = {k: v[0] for k, v in SETTINGS.items()}
    s.update(_saved())
    s.update({k: v for k, v in overrides.items() if v is not None})
    if s["threads"] in (None, "", 0):
        s["threads"] = cpu_count()
    s["frames"] = _frames(s["frames"])
    t = float(s["threshold"])
    if not 0 <= t <= 255:
        raise ValueError("threshold must be 0 to 1, or 0 to 255 on the image scale")
    s["threshold"] = round(t / 255, 4) if t > 1 else t
    if int(s["augspeed"]) not in (1, 2, 4, 10):
        raise ValueError("augspeed must be 1, 2, 4 or 10")
    if s["device"] not in ("auto", "gpu", "cpu"):
        raise ValueError("device must be auto, gpu or cpu")
    return s


def _frames(f):
    fl = [x.strip() for x in (f.split(",") if isinstance(f, str) else f) if x.strip()]
    bad = [x for x in fl if x not in FRAMES]
    if bad or not fl:
        raise ValueError("frames must be from 1fm, 3fm, 5fm (got %s)" % f)
    return [x for x in FRAMES if x in fl]


def _settings_table(s, keys):
    rows = ["| Setting | Value |", "|---|---|"]
    for k in keys:
        v = s[k]
        rows.append("| %s | %s |" % (k, ", ".join(v) if isinstance(v, list) else v))
    return "\n".join(rows)


def settings():
    """Print every setting with its current value, default and meaning."""
    saved = _saved()
    s = _effective({})
    print("## Settings\n")
    print("Any setting can be changed for a single run, or saved as this project's default.\n")
    print("| Setting | Current | Default | Meaning |")
    print("|---|---|---|---|")
    for k, (d, doc) in SETTINGS.items():
        cur = s[k]
        cur = ", ".join(cur) if isinstance(cur, list) else cur
        mark = " (project)" if k in saved else ""
        print("| %s | %s%s | %s | %s |" % (k, cur, mark, "—" if d is None else d, doc))
    print("\nProject settings file: `%s`" % _settings_file())
    return s


def set_defaults(**kw):
    """Change settings for this project (saved in the project folder). Pass None to restore a default."""
    _effective({k: v for k, v in kw.items() if v is not None})          # validate
    saved = _saved()
    for k, v in kw.items():
        if v is None:
            saved.pop(k, None)
        else:
            saved[k] = v
    json.dump(saved, open(_settings_file(), "w"), indent=1)
    print("Saved for this project: " + (", ".join("%s = %s" % kv for kv in saved.items()) or "nothing (all defaults)"))
    return saved


# ----------------------------------------------------------------------------- hardware
def cpu_count():
    """CPUs this container may use (its CPU limit, not the machine's core count)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(int(q) / int(p))))
    except (OSError, ValueError):
        pass
    return n


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def _seconds_per_update(s, gpu):
    # Measured: 0.2 s per batch on an RTX 2080 Ti; ~3.2 s per batch on 8 CPU threads (fewer threads are
    # slower in proportion; more than 8 were not measured, so no further gain is assumed).
    per_batch = 0.2 if (gpu and s["device"] != "cpu") else 3.2 * 8 / min(8, max(1, int(s["threads"])))
    return per_batch * int(s["iter_size"]) * int(s["batch"]) / 5


def _fmt_time(sec):
    sec = int(sec)
    if sec < 90:
        return "%d s" % sec
    if sec < 5400:
        return "%d min" % round(sec / 60)
    return "%.1f h" % (sec / 3600)


def environment():
    """Report the hardware, the folders used and rough time estimates."""
    gpu = _gpu()
    s = _effective({})
    try:
        import torch
        cuda = torch.cuda.is_available()
        tv = torch.__version__
    except Exception:
        cuda, tv = False, "not installed"
    print("## Environment\n")
    print("| | |\n|---|---|")
    print("| GPU | %s |" % (gpu or "none"))
    print("| PyTorch | %s, GPU training %s |" % (tv, "available" if cuda else "not available"))
    print("| CPUs available | %d |" % cpu_count())
    print("| Project folder (kept): own models, predictions, settings | `%s`, %.1f GB free |" % (_workspace(), _free_gb(_workspace())))
    print("| Temporary folder: pretrained models, training data, training runs | `%s`, %.1f GB free, not kept when the pod stops |"
          % (_scratch(), _free_gb(_scratch())))
    spu = _seconds_per_update(s, gpu if cuda else None)
    print("\nTraining with the current settings: about %.1f s per update, so %d updates take about %s. "
          "Estimates from measured runs; the first updates show the real speed."
          % (spu, s["updates"], _fmt_time(spu * s["updates"])))
    return {"gpu": gpu, "cuda": cuda, "cpus": cpu_count(), "workspace": _workspace()}


# ----------------------------------------------------------------------------- data helpers
def sample_data():
    """Paths of the sample dataset shipped with CDeep3M2 (SBEM mitochondria, 1024x1024 slices)."""
    root = os.path.join(CDEEP3M2, "mito_testsample")
    d = {"training_images": root + "/training/images", "training_labels": root + "/training/labels",
         "validation_images": root + "/validation/images", "validation_labels": root + "/validation/labels",
         "test_images": root + "/testset"}
    print("## Sample data (SBEM mitochondria)\n")
    print("| Set | Folder | Images |\n|---|---|---|")
    for k, v in d.items():
        print("| %s | `%s` | %d |" % (k.replace("_", " "), v, len(_images(v))))
    return d


def _images(folder):
    if os.path.isfile(folder):
        return [folder]
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMAGE_EXT))


def _check_folder(folder, what):
    files = _images(folder)
    if not files:
        raise FileNotFoundError("no images (.png/.tif) found for %s: %s" % (what, folder))
    return files


def _run(cmd, log, cwd=None, env=None):
    with open(log, "a") as f:
        f.write("\n$ %s\n" % " ".join(map(str, cmd)))
        f.flush()
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def _tail(path, n=15):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- models
def _mb(n):
    return "%.0f MB" % (n / 1e6) if n else "?"


def _targets(z, target):
    """True when a model segments the target; matched on the word's first letters, so "mitochondrion",
    "mitochondria", "membrane" and "membranes" all match."""
    if not target:
        return True
    return any(w[:6] in z["target"] for w in re.findall(r"[a-z]+", str(target).lower()) if len(w) > 2)


def models(target=None, imaging=None, include_tests=False):
    """List the pretrained models of the CDeep3M model zoo (optionally only those for a target such as
    "mitochondria" or "membranes", or an imaging method such as "SBEM") and the project's own models."""
    zoo = [z for z in ZOO.values() if (include_tests or not z.get("test_entry"))
           and _targets(z, target)
           and (not imaging or str(imaging).lower() in z["imaging"].lower())]
    print("## Models\n")
    print("Pretrained models from the CDeep3M model zoo, by name or CIL model ID. Each was trained on images "
          "of one pixel size; images at a similar size segment best.\n")
    print("| Model | CIL ID | Segments | Imaging | Pixel size (X/Y, Z) | Networks | Size | Downloaded |")
    print("|---|---|---|---|---|---|---|---|")
    for z in zoo:
        px = "%g nm, %s" % (z["pixel_nm"], "%g nm" % z["slice_nm"] if z.get("slice_nm") else "Z not stated") \
            if z.get("pixel_nm") else "not stated"
        here = os.path.isdir(os.path.join(_zoo_dir(), z["name"], "1fm"))
        print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            z["name"], z["id"], z["about"], z["imaging"], px, ", ".join(z.get("networks") or []) or "?",
            _mb(z.get("bytes")), "yes" if here else "no (on first use)"))
    own = []
    for d in sorted(glob.glob(os.path.join(_models_dir(), "*"))):
        name = os.path.basename(d)
        if name in ZOO or not os.path.isdir(d):
            continue
        info = {}
        try:
            info = json.load(open(os.path.join(d, "source.json")))
        except (OSError, ValueError):
            pass
        own.append(name)
        if len(own) == 1:
            print("\n**Trained in this project**\n\n| Model | Trained from | Updates | Networks | Folder |\n|---|---|---|---|---|")
        frames = [f for f in FRAMES if os.path.isdir(os.path.join(d, f))]
        print("| %s | %s | %s | %s | `%s` |" % (name, info.get("base_model", "?"), info.get("updates", "?"),
                                               ", ".join(frames), d))
    print("\nCite the model zoo as: Haberl et al., Nat Methods 15, 677-680 (2018), doi:10.1038/s41592-018-0106-z.")
    return {"pretrained": [z["name"] for z in zoo], "own": own}


_IMAGING = {"SBEM": ("scanning electron", "sem", "block"), "TEM": ("transmission electron", "tem", "electron; film"),
            "electron tomography": ("tomograph",)}


def _fit(z, pixel_nm=None, imaging=None):
    """How well a pretrained model suits data of this pixel size and imaging: (rating, note)."""
    notes, rating = [], "unknown"
    if pixel_nm and z.get("pixel_nm"):
        r = float(pixel_nm) / z["pixel_nm"]
        rating = "good" if 1 / 1.5 <= r <= 1.5 else "fair" if 1 / 2.5 <= r <= 2.5 else "poor"
        notes.append("pixels %g nm vs the model's %g nm (%s)" % (
            pixel_nm, z["pixel_nm"], "similar" if rating == "good" else
            "the images are %.1f× %s" % (max(r, 1 / r), "coarser" if r > 1 else "finer")))
    elif z.get("pixel_nm"):
        notes.append("pixel size of the images not known; the model's is %g nm" % z["pixel_nm"])
    if imaging:
        im = imaging.lower()
        keys = [k for k, words in _IMAGING.items() if any(w in im for w in words)]
        if keys:
            same = any(k.lower() in z["imaging"].lower() for k in keys)
            notes.append("imaging %s the model's (%s)" % ("matches" if same else "differs from", z["imaging"]))
            if not same and rating == "good":
                rating = "fair"
        elif "electron" in im:
            notes.append("electron microscopy, type not stated")
        else:
            notes.append("not electron microscopy (the models are trained on EM)")
            rating = "poor"
    return rating, "; ".join(notes)


def _data_info(images):
    """(pixel_nm, imaging, label) of the data: from a CIL record's catalog entry, else unknown."""
    if isinstance(images, dict) and images.get("folder"):          # what cil.slices() returned
        if images.get("id") is not None:
            return (images.get("pixel_nm") or [None])[0], images.get("imaging"), "CIL_%s" % images["id"]
        return None, None, images["folder"]
    if _is_cil(images):
        try:
            r = _cil()._record_or_fail(images)
            return r.get("pixel_nm_x"), r.get("imaging_mode"), "CIL_%d" % r["id"]
        except LookupError:
            return None, None, str(images)
    return None, None, str(images)


def suggest_models(images=None, target=None, pixel_nm=None, imaging=None):
    """Rank the pretrained models for some data: a CIL record ID (its pixel size and imaging are read from
    the catalog) or a folder, with pixel_nm and imaging given when known. target: e.g. "mitochondria"."""
    px, im, label = _data_info(images) if images is not None else (None, None, "the images")
    px, im = pixel_nm or px, imaging or im
    zoo = [z for z in ZOO.values() if not z.get("test_entry")
           and _targets(z, target)]
    order = {"good": 0, "fair": 1, "unknown": 2, "poor": 3}
    ranked = sorted(((z,) + _fit(z, px, im) for z in zoo),
                    key=lambda t: (order[t[1]], abs((px or t[0].get("pixel_nm") or 1) - (t[0].get("pixel_nm") or 0))))
    print("## Models for %s\n" % label)
    print("Data: pixel size %s, imaging %s%s.\n" % ("%g nm" % px if px else "not known", im or "not stated",
                                                    ", as recorded in the CIL catalog" if _is_cil(images) else ""))
    print("| Model | CIL ID | Segments | Fit | Why |\n|---|---|---|---|---|")
    for z, rating, note in ranked:
        print("| %s | %s | %s | %s | %s |" % (z["name"], z["id"], z["about"], rating, note))
    print("\nCDeep3M models work best on images at the pixel size they were trained on; a poor fit can "
          "still be tested on a few images and scored, or improved by fine-tuning.")
    return {"data": label, "pixel_nm": px, "imaging": im,
            "models": [{"model": z["name"], "id": z["id"], "fit": r, "why": n} for z, r, n in ranked]}


def _is_cil(x):
    if isinstance(x, dict):
        return "id" in x and "folder" not in x
    if isinstance(x, int):
        return True
    return isinstance(x, str) and not os.path.exists(x) and bool(re.fullmatch(r"\s*(?:CIL[_: ]?)?\d+\s*", x, re.I))


def _cil():
    """The cil skill's library (the Cell Image Library), installed next to this skill."""
    try:
        import cil_lib
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        for d in (os.path.join(os.path.dirname(here), "cil"), os.path.expanduser("~/.deepagents/agent/skills/cil")):
            if os.path.isfile(os.path.join(d, "cil_lib.py")) and d not in sys.path:
                sys.path.insert(0, d)
        import cil_lib
    return cil_lib


def _caffemodel(frame_dir):
    cands = glob.glob(os.path.join(frame_dir, "trainedmodel", "*.caffemodel"))
    if not cands:
        raise FileNotFoundError("no .caffemodel in %s/trainedmodel" % frame_dir)
    num = lambda p: int((re.findall(r"(\d+)", os.path.basename(p)) or ["0"])[-1])
    return sorted(cands, key=num)[-1]


def get_model(name, frames=None, calibration_images=None):
    """Make a model ready for prediction: download a pretrained one if needed, convert to ncnn.
    calibration_images: a folder of training images to also build an int8 version (faster on the CPU)."""
    name = _model_key(name)
    d = _model_path(name)
    fl = _frames(frames or "1fm")
    log = os.path.join(_scratch(), "get_model.log")
    if name in ZOO and ZOO[name].get("networks"):
        missing = [f for f in fl if f not in ZOO[name]["networks"]]
        if missing:
            raise ValueError("model %s has only the %s network(s), not %s" % (
                name, ", ".join(ZOO[name]["networks"]), ", ".join(missing)))
    if not os.path.isfile(os.path.join(d, "1fm", "deploy.prototxt")):
        if name not in ZOO:
            raise FileNotFoundError("unknown model %r; models() lists the pretrained and own models" % name)
        z = ZOO[name]
        tmp = os.path.join(_scratch(), "download_" + name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        print("Downloading %s (CIL %s, %s) from the CDeep3M model zoo ..." % (z["title"], z["id"], _mb(z.get("bytes"))),
              flush=True)
        t0 = time.time()
        urllib.request.urlretrieve(z["url"], os.path.join(tmp, "model.tar"))
        with tarfile.open(os.path.join(tmp, "model.tar")) as t:
            t.extractall(tmp, filter="data")
        hits = glob.glob(os.path.join(tmp, "**", "1fm", "deploy.prototxt"), recursive=True)
        if not hits:
            raise RuntimeError("the downloaded archive has no 1fm/deploy.prototxt")
        shutil.rmtree(d, ignore_errors=True)
        shutil.move(os.path.dirname(os.path.dirname(hits[0])), d)
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump({"source": "model zoo", "id": z["id"], "title": z["title"], "doi": z["doi"], "url": z["url"]},
                  open(os.path.join(d, "source.json"), "w"), indent=1)
        print("Downloaded in %s." % _fmt_time(time.time() - t0), flush=True)
    done = []
    for f in fl:
        fd = os.path.join(d, f)
        if not os.path.isdir(fd):
            raise FileNotFoundError("model %s has no %s network" % (name, f))
        need32 = not os.path.isfile(os.path.join(fd, "ncnn", "model.param"))
        need8 = calibration_images and not os.path.isfile(os.path.join(fd, "ncnn", "int8", "model.param"))
        if not (need32 or need8):
            continue
        cmd = [ANYWHERE + "/ncnn/convert_model.sh", os.path.join(fd, "deploy.prototxt"), _caffemodel(fd),
               os.path.join(fd, "ncnn")]
        if need8:
            cal = os.path.join(_scratch(), "calib_%s_%s" % (name, f))
            shutil.rmtree(cal, ignore_errors=True)
            if _run(["python3", ANYWHERE + "/ncnn/make_calibration.py", "--repo", CDEEP3M2, "--images",
                     calibration_images, "--out", cal, "--model", f], log) != 0:
                raise RuntimeError("int8 calibration failed:\n" + _tail(log))
            cmd.append(os.path.join(cal, "list.txt"))
        if _run(cmd, log) != 0:
            raise RuntimeError("model conversion failed:\n" + _tail(log))
        done.append(f + (" (with int8)" if need8 else ""))
    print("Model **%s** ready for prediction (%s)%s." % (
        name, ", ".join(fl), "; converted now: " + ", ".join(done) if done else ""))
    return {"model": name, "folder": d, "frames": fl}


# ----------------------------------------------------------------------------- prediction
def _stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def predict(images, model="sbem-mitochondria", name=None, **overrides):
    """Segment the images in a folder (or one image). Settings: frames, augspeed, device, precision, threshold."""
    model = _model_key(model)
    s = _effective(overrides)
    px, imaging, source = _data_info(images)
    if isinstance(images, dict) and images.get("folder"):
        images = images["folder"]
    elif _is_cil(images):
        got = _cil().slices(images)
        if not got.get("ready"):
            return {"name": None, "ready": False, "reason": "the record's images are large; confirm the download first"}
        images = got["folder"]
    files = _check_folder(images, "prediction")
    name = name or "%s_%s" % (model, _stamp())
    gpu = _gpu()
    print("## Prediction: %s\n" % name)
    print("**Settings used**\n")
    print(_settings_table(dict(s, model=model, images=source if source != str(images) else images),
                          ["model", "images", "frames", "augspeed", "device", "precision", "threshold"]))
    if model in ZOO and (px or imaging):
        rating, note = _fit(ZOO[model], px, imaging)
        print("\nModel fit for these images: **%s** (%s)." % (rating, note))
    print(flush=True)
    get_model(model, frames=s["frames"])
    work = os.path.join(_scratch(), "predict", name)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.dirname(work), exist_ok=True)
    log = work + ".log"
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ)
    if s["precision"] != "auto":
        env["NCNN_PRECISION"] = s["precision"]
    dev = {"auto": "auto", "gpu": "vulkan", "cpu": "cpu"}[s["device"]]
    t0 = time.time()
    rc = _run([CDEEP3M2 + "/runprediction.sh", "--models", ",".join(s["frames"]), "--denoise", "0",
               "--augspeed", s["augspeed"], "--backend", "ncnn", "--device", dev,
               _model_path(model), images, work], log, cwd=CDEEP3M2, env=env)
    took = time.time() - t0
    if rc != 0 or not os.path.isfile(os.path.join(work, "DONE")):
        err = _tail(os.path.join(work, "ERROR"), 5) + _tail(log)
        raise RuntimeError("prediction failed (exit %s):\n%s" % (rc, err))
    out = os.path.join(_workspace(), "predictions", name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    probs = sorted(glob.glob(os.path.join(work, "ensembled", "Segmented_*.png")))
    shutil.copytree(os.path.join(work, "ensembled"), os.path.join(out, "probability"), dirs_exist_ok=True)
    if os.path.isdir(os.path.join(work, "overlay")):
        shutil.copytree(os.path.join(work, "overlay"), os.path.join(out, "overlay"))
    masks = _write_masks(os.path.join(out, "probability"), os.path.join(out, "mask"), s["threshold"])
    used = re.findall(r"[Rr]unning prediction with ncnn on ([^\n(]+)", _tail(os.path.join(work, "logs", "prediction.log"), 400))
    meta = {"name": name, "model": model, "images": os.path.abspath(images), "source": source, "image_files": [os.path.basename(f) for f in files],
            "settings": {k: s[k] for k in ("frames", "augspeed", "device", "precision", "threshold")},
            "seconds": round(took, 1), "device_used": used[-1].strip() if used else dev, "gpu": gpu}
    json.dump(meta, open(os.path.join(out, "prediction.json"), "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    print("**Result**\n")
    print("| | |\n|---|---|")
    print("| Images segmented | %d |" % len(probs))
    print("| Time | %s (%s per image) |" % (_fmt_time(took), _fmt_time(took / max(1, len(probs)))))
    print("| Ran on | %s |" % meta["device_used"])
    print("| Probability maps (0–255) | `%s/probability` |" % out)
    print("| Masks (probability > %s) | `%s/mask` |" % (s["threshold"], out))
    print("| Overlays on the images | `%s/overlay` |" % out)
    print("\nPrediction name: `%s`" % name)
    _saved_line(out)
    return {"name": name, "folder": out, "probability": probs, "masks": masks, "seconds": took}


def _write_masks(prob_dir, mask_dir, threshold):
    from PIL import Image
    os.makedirs(mask_dir, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(prob_dir, "Segmented_*.png"))):
        a = np.array(Image.open(p).convert("L"))
        m = os.path.join(mask_dir, os.path.basename(p).replace("Segmented_", "Mask_"))
        Image.fromarray(((a > threshold * 255) * 255).astype(np.uint8)).save(m)
        out.append(m)
    return out


def _name(x):
    """Functions accept a name or the dict an earlier step returned (predict, prepare_training, train)."""
    return x["name"] if isinstance(x, dict) and "name" in x else x


def _prediction_folder(prediction):
    prediction = _name(prediction)
    if os.path.isdir(prediction):
        return prediction
    d = os.path.join(_workspace(), "predictions", prediction)
    if not os.path.isdir(d):
        raise FileNotFoundError("no prediction %r in %s" % (prediction, os.path.dirname(d)))
    return d


# ----------------------------------------------------------------------------- evaluation and figures
def _pairs(prediction, labels):
    from PIL import Image
    d = _prediction_folder(prediction)
    probs = sorted(glob.glob(os.path.join(d, "probability", "Segmented_*.png")))
    labs = _check_folder(labels, "labels")
    if len(labs) < len(probs):
        raise ValueError("%d predictions but only %d label images" % (len(probs), len(labs)))
    meta = {}
    try:
        meta = json.load(open(os.path.join(d, "prediction.json")))
    except (OSError, ValueError):
        pass
    names = meta.get("image_files") or [os.path.basename(p) for p in probs]
    out = []
    for i, p in enumerate(probs):
        pr = np.array(Image.open(p).convert("L"))
        lb = np.array(Image.open(labs[i]).convert("L")) > 0
        if pr.shape != lb.shape:
            raise ValueError("size differs: %s %s vs %s %s" % (os.path.basename(p), pr.shape, labs[i], lb.shape))
        out.append((names[i] if i < len(names) else os.path.basename(p), os.path.basename(labs[i]), pr, lb))
    return d, meta, out


def evaluate(prediction, labels, threshold=None):
    """Score a prediction against expert labels (sorted order pairs the files): Dice, IoU, precision, recall."""
    s = _effective({"threshold": threshold})
    d, meta, pairs = _pairs(prediction, labels)
    t = s["threshold"]
    print("## Evaluation: %s\n" % os.path.basename(d))
    print("Threshold %s (a pixel is the object when its probability is above it); labels from `%s`, paired "
          "with the images in sorted order.\n" % (t, labels))
    rows, TP, FP, FN = [], 0, 0, 0
    print("| Image | Label | Dice | IoU | Precision | Recall |\n|---|---|---|---|---|---|")
    for img, lab, pr, lb in pairs:
        m = pr > t * 255
        tp, fp, fn = int((m & lb).sum()), int((m & ~lb).sum()), int((~m & lb).sum())
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        r = _scores(tp, fp, fn)
        rows.append(dict(image=img, label=lab, tp=tp, fp=fp, fn=fn, **r))
        print("| %s | %s | %.4f | %.4f | %.4f | %.4f |" % (img, lab, r["dice"], r["iou"], r["precision"], r["recall"]))
    tot = _scores(TP, FP, FN)
    print("| **All %d images** | | **%.4f** | **%.4f** | **%.4f** | **%.4f** |" % (
        len(rows), tot["dice"], tot["iou"], tot["precision"], tot["recall"]))
    print("\nOverall scores pool the pixels of all images; the mean of the per-image Dice is %.4f." %
          float(np.mean([r["dice"] for r in rows])))
    path = os.path.join(d, "evaluation.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Scores saved to `%s`." % path)
    return {"per_image": rows, "overall": tot, "csv": path}


def compare(predictions, labels, threshold=None):
    """Compare two or more predictions of the same images against the same labels: overall scores side
    by side, per-image Dice with the change from the first, how many images improved, and a bar chart.
    The first prediction is the reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _effective({"threshold": threshold})
    t = s["threshold"]
    preds = [_name(p) for p in predictions]
    if len(preds) < 2:
        raise ValueError("compare needs at least two predictions")
    res = {}
    for p in preds:
        d, meta, pairs = _pairs(p, labels)
        per, TP, FP, FN = [], 0, 0, 0
        for img, lab, pr, lb in pairs:
            m = pr > t * 255
            tp, fp, fn = int((m & lb).sum()), int((m & ~lb).sum()), int((~m & lb).sum())
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            per.append((img, _scores(tp, fp, fn)["dice"]))
        res[p] = {"model": meta.get("model", "?"), "overall": _scores(TP, FP, FN), "per_image": per,
                  "fp": FP, "fn": FN}
    ref = preds[0]
    imgs = [i for i, _ in res[ref]["per_image"]]
    for p in preds[1:]:
        if [i for i, _ in res[p]["per_image"]] != imgs:
            raise ValueError("%s and %s were not made from the same images" % (ref, p))
    print("## Comparison (reference: %s)\n" % ref)
    print("Threshold %s; labels from `%s`; overall scores pool the pixels of all %d images.\n" % (t, labels, len(imgs)))
    print("| Prediction | Model | Dice | IoU | Precision | Recall | Pixels predicted wrongly | Pixels missed |")
    print("|---|---|---|---|---|---|---|---|")
    for p in preds:
        o = res[p]["overall"]
        print("| %s | %s | %.4f | %.4f | %.4f | %.4f | %d | %d |" % (p, res[p]["model"], o["dice"], o["iou"],
                                                               o["precision"], o["recall"], res[p]["fp"], res[p]["fn"]))
    for p in preds[1:]:
        o, r = res[p]["overall"], res[ref]["overall"]
        print("| change, %s vs %s | | %+.4f | %+.4f | %+.4f | %+.4f | %+d | %+d |" % (
            p, ref, o["dice"] - r["dice"], o["iou"] - r["iou"], o["precision"] - r["precision"],
            o["recall"] - r["recall"], res[p]["fp"] - res[ref]["fp"], res[p]["fn"] - res[ref]["fn"]))
    print("\n**Dice per image**\n")
    print("| Image | " + " | ".join(preds) + " | " + " | ".join("change, %s" % p for p in preds[1:]) + " |")
    print("|---|" + "---|" * (2 * len(preds) - 1))
    for k, img in enumerate(imgs):
        vals = [res[p]["per_image"][k][1] for p in preds]
        print("| %s | %s | %s |" % (img, " | ".join("%.4f" % v for v in vals),
                                    " | ".join("%+.4f" % (v - vals[0]) for v in vals[1:])))
    for p in preds[1:]:
        diffs = [res[p]["per_image"][k][1] - res[ref]["per_image"][k][1] for k in range(len(imgs))]
        up, down = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
        print("\n%s against %s: Dice higher on %d of %d images, lower on %d, unchanged on %d." % (
            p, ref, up, len(imgs), down, len(imgs) - up - down))
    names = ["dice", "iou", "precision", "recall"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, k in zip(axes, names):
        vals = [res[p]["overall"][k] for p in preds]
        bars = ax.bar(range(len(preds)), vals, color=["#1f4e79", "#c0504d", "#4f8f3a", "#8064a2"][:len(preds)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.4f" % v, ha="center", fontsize=9)
        ax.set_xticks(range(len(preds)))
        ax.set_xticklabels(preds, rotation=15, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title({"iou": "IoU"}.get(k, k.capitalize()))
    fig.suptitle("Overall scores, threshold %s" % t)
    fig.tight_layout()
    path = os.path.join(_prediction_folder(preds[-1]), "comparison_scores.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    shown = _for_display(path)
    print("\nChart (embed this path): `%s`" % shown)
    return {"predictions": res, "figure": shown}


def _scores(tp, fp, fn):
    return {"dice": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}


def show(prediction, labels=None, n=3, threshold=None, filename=None):
    """Figure of the first n images: image, probability, mask, and with labels the expert label and an
    agreement map (green both, red predicted only, blue label only). Returns the PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    s = _effective({"threshold": threshold})
    t = s["threshold"]
    d = _prediction_folder(prediction)
    meta = json.load(open(os.path.join(d, "prediction.json")))
    imgs = _images(meta["images"])
    probs = sorted(glob.glob(os.path.join(d, "probability", "Segmented_*.png")))[:n]
    labs = _check_folder(labels, "labels")[:n] if labels else None
    cols = ["Image", "Probability", "Mask (p > %s)" % t] + (["Expert label", "Agreement"] if labs else [])
    fig, axes = plt.subplots(len(probs), len(cols), figsize=(3.2 * len(cols), 3.3 * len(probs)), squeeze=False)
    for i, p in enumerate(probs):
        pr = np.array(Image.open(p).convert("L"))
        m = pr > t * 255
        panels = [np.array(Image.open(imgs[i]).convert("L")), pr, m]
        if labs:
            lb = np.array(Image.open(labs[i]).convert("L")) > 0
            ag = np.zeros(m.shape + (3,), np.uint8)
            ag[m & lb] = (40, 170, 70)
            ag[m & ~lb] = (220, 50, 40)
            ag[~m & lb] = (40, 90, 220)
            panels += [lb, ag]
        for j, im in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=11)
        axes[i][0].set_ylabel(os.path.basename(imgs[i]), fontsize=9)
    if labs:
        fig.legend(handles=[Patch(color=(40/255, 170/255, 70/255), label="both"),
                            Patch(color=(220/255, 50/255, 40/255), label="predicted only"),
                            Patch(color=(40/255, 90/255, 220/255), label="label only")],
                   loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04 if labs else 0, 1, 1))
    path = os.path.join(d, filename or "comparison.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    shown = _for_display(path)
    print("Figure (embed this path): `%s`" % shown)
    return shown


# ----------------------------------------------------------------------------- training
def prepare_training(images, labels, name=None):
    """Turn labelled slices (images + matching binary label images) into training data."""
    from PIL import Image
    im, lb = _check_folder(images, "training images"), _check_folder(labels, "training labels")
    if len(im) != len(lb):
        raise ValueError("%d images but %d labels" % (len(im), len(lb)))
    s0, s1 = Image.open(im[0]).size, Image.open(lb[0]).size
    if s0 != s1:
        raise ValueError("image size %s differs from label size %s" % (s0, s1))
    name = name or "data_" + _stamp()
    out = os.path.join(_scratch(), "training_data", name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    log = out + ".log"
    t0 = time.time()
    if _run(["python3", "PreprocessTrainingData.py", images, labels, "0", out], log, cwd=CDEEP3M2) != 0 \
            or not glob.glob(os.path.join(out, "*.h5")):
        raise RuntimeError("training data preparation failed:\n" + _tail(log))
    size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(out, "*"))) / 1e9
    json.dump({"images": os.path.abspath(images), "labels": os.path.abspath(labels), "slices": len(im)},
              open(os.path.join(out, "source.json"), "w"), indent=1)
    print("## Training data: %s\n" % name)
    print("| | |\n|---|---|")
    print("| Labelled slices | %d (%d x %d) |" % (len(im), s0[0], s0[1]))
    print("| Augmented training files | %.1f GB in `%s` |" % (size, out))
    print("| Time | %s |" % _fmt_time(time.time() - t0))
    print("\nTraining data name: `%s` (temporary: kept only while the pod runs)" % name)
    return {"name": name, "folder": out}


def _data_folder(data):
    data = _name(data)
    if os.path.isdir(data) and glob.glob(os.path.join(data, "*.h5")):
        return data
    d = os.path.join(_scratch(), "training_data", data)
    if not glob.glob(os.path.join(d, "*.h5")):
        raise FileNotFoundError("no training data %r; run prepare_training() first (it is lost when the pod stops)" % data)
    return d


def _run_dir(name):
    """Training runs (checkpoints of ~200 MB each) stay in the temporary folder with their training
    data: a run cannot continue without that data anyway. finish_training() keeps the result."""
    return os.path.join(_scratch(), "runs", name)


def _alive(pid):
    """True while the training launcher runs. A finished child of this process stays a zombie until it is
    reaped, so reap it when it is ours; otherwise read its state, and check it is still our train.sh
    (after a pod restart the number can belong to another process)."""
    if not pid:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        if open("/proc/%d/stat" % pid).read().rsplit(")", 1)[1].split()[0] in ("Z", "X"):
            return False
        return b"train.sh" in open("/proc/%d/cmdline" % pid, "rb").read()
    except (OSError, IndexError):
        return False


def train(model, training_data, name=None, start=False, **overrides):
    """Fine-tune a model (or train from scratch with model=None) in the background.
    Without start=True it only shows the plan. Settings: frames, updates, learning_rate, iter_size, batch,
    device, threads, checkpoint_every. Running it again with the same name continues a stopped run."""
    model = _model_key(model)
    from_scratch = model is None
    if from_scratch and "learning_rate" not in overrides and "learning_rate" not in _saved():
        overrides["learning_rate"] = 0.01
    s = _effective(overrides)
    data = _data_folder(training_data)
    base = model or "sbem-mitochondria"          # export needs a model of the same layout as a template
    name = name or "run_" + _stamp()
    rd = _run_dir(name)
    gpu = _gpu()
    spu = _seconds_per_update(s, gpu if s["device"] != "cpu" else None)
    total = spu * s["updates"] * len(s["frames"])
    print("## Training %s: %s\n" % ("plan" if not start else "started", name))
    print("**Settings used**\n")
    print(_settings_table(dict(s, model="none (from scratch)" if from_scratch else model, data=_name(training_data)),
                          ["model", "data", "frames", "updates", "learning_rate", "iter_size", "batch",
                           "device", "threads", "checkpoint_every"]))
    print("\nHardware: %s, %d CPUs. Estimated time: about %s (%.1f s per update%s)." % (
        gpu or "no GPU", s["threads"], _fmt_time(total), spu,
        ", networks trained one after another" if len(s["frames"]) > 1 else ""))
    info = {}
    try:
        info = json.load(open(os.path.join(rd, "run.json")))
    except (OSError, ValueError):
        pass
    if info and _alive(info.get("pid")):
        print("\nThis run is already training (process %s); ask how it is going at any time." % info["pid"])
        return {"name": name, "running": True}
    if not start:
        print("\nNothing has started yet; training starts only when you confirm.")
        return {"name": name, "started": False, "estimated_seconds": total}
    get_model(base, frames=s["frames"])
    os.makedirs(rd, exist_ok=True)
    if os.path.exists(os.path.join(rd, "STOP")):
        os.remove(os.path.join(rd, "STOP"))
    dev = {"auto": "auto", "gpu": "cuda", "cpu": "cpu"}[s["device"]]
    lines = ["cd %s" % ANYWHERE]
    for f in s["frames"]:
        fd = os.path.join(_model_path(base), f)
        out = os.path.join(rd, f)
        args = ["python3", "-m", "trainer.train", "--model", f, "--deploy", os.path.join(fd, "deploy.prototxt"),
                "--data", data, "--out", out, "--max-iter", s["updates"], "--base-lr", s["learning_rate"],
                "--iter-size", s["iter_size"], "--batch", s["batch"], "--threads", s["threads"],
                "--device", dev, "--checkpoint-every", s["checkpoint_every"], "--keep", 1,
                "--progress-dir", os.path.join(out, "progress")]
        if not from_scratch:
            args += ["--weights", _caffemodel(fd)]
        lines += ['[ -f "%s/STOP" ] && exit 0' % rd, "mkdir -p %s" % out,
                  " ".join('"%s"' % a for a in args) + ' >> "%s/train.log" 2>&1 || exit 1' % out]
    lines.append('echo done > "%s/FINISHED"' % rd)
    script = os.path.join(rd, "train.sh")
    open(script, "w").write("\n".join(lines) + "\n")
    if os.path.exists(os.path.join(rd, "FINISHED")):
        os.remove(os.path.join(rd, "FINISHED"))
    p = subprocess.Popen(["bash", script], start_new_session=True, stdout=open(os.path.join(rd, "launcher.log"), "a"),
                         stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH=ANYWHERE))
    info = {"name": name, "pid": p.pid, "base_model": base, "from_scratch": from_scratch, "data": data,
            "settings": {k: s[k] for k in ("frames", "updates", "learning_rate", "iter_size", "batch", "device",
                                           "threads", "checkpoint_every")},
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": gpu}
    json.dump(info, open(os.path.join(rd, "run.json"), "w"), indent=1)
    print("\nStarted in the background (process %d). Run name: `%s`. It keeps running between questions; "
          "ask how the training is going at any time." % (p.pid, name))
    return {"name": name, "started": True, "pid": p.pid, "folder": rd}


def training_status(name, plot=True):
    """Progress of a training run: updates done, loss, speed, time left, and a loss curve."""
    name = _name(name)
    rd = _run_dir(name)
    try:
        info = json.load(open(os.path.join(rd, "run.json")))
    except (OSError, ValueError):
        raise FileNotFoundError("no training run %r in %s" % (name, os.path.dirname(rd)))
    running = _alive(info.get("pid"))
    finished = os.path.exists(os.path.join(rd, "FINISHED"))
    stopped = os.path.exists(os.path.join(rd, "STOP"))
    state = "finished" if finished else "running" if running else "stopped" if stopped else "not running (failed or interrupted)"
    print("## Training status: %s — %s\n" % (name, state))
    print("| Network | Updates | Last loss | Average loss | Speed | Time left |\n|---|---|---|---|---|---|")
    curves, frames_out = {}, []
    for f in info["settings"]["frames"]:
        prog = os.path.join(rd, f, "progress")
        st = {}
        for j in glob.glob(os.path.join(prog, "*.json")):
            st = json.load(open(j))
        rows = []
        for c in glob.glob(os.path.join(prog, "*.csv")):
            rows = list(csv.DictReader(open(c)))
        if rows:
            curves[f] = ([int(r["update"]) for r in rows], [float(r["loss_reported"]) for r in rows])
        done = int(rows[-1]["update"]) if rows else int(st.get("update", 0))
        total = info["settings"]["updates"]
        spu = float(np.median([float(r["seconds_per_update"]) for r in rows[-50:]])) if rows else None
        avg = float(np.mean(curves[f][1][-16:])) if rows else None
        left = _fmt_time(spu * (total - done)) if (spu and done < total and running) else "—"
        print("| %s | %d of %d | %s | %s | %s | %s |" % (
            f, done, total, "%.4f" % curves[f][1][-1] if rows else "—", "%.4f" % avg if avg else "—",
            "%.1f s/update" % spu if spu else "—", left))
        frames_out.append({"frame": f, "updates": done, "of": total, "seconds_per_update": spu, "avg_loss": avg})
    if state.startswith("not running"):
        for f in info["settings"]["frames"]:
            t = _tail(os.path.join(rd, f, "train.log"), 8)
            if t:
                print("\nEnd of the %s log:\n```\n%s```" % (f, t))
    fig = None
    if plot and curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 3.5))
        for f, (x, y) in curves.items():
            k = max(1, min(16, len(y)))
            plt.plot(x, y, alpha=0.3)
            plt.plot(x[k - 1:], np.convolve(y, np.ones(k) / k, mode="valid"), label="%s (average of %d)" % (f, k))
        plt.xlabel("update"); plt.ylabel("training loss"); plt.legend(); plt.title("Training loss: " + name)
        plt.tight_layout()
        fig = os.path.join(rd, "loss.png")
        plt.savefig(fig, dpi=110)
        plt.close()
        fig = _for_display(fig)
        print("\nLoss curve (embed this path): `%s`" % fig)
    return {"name": name, "state": state, "networks": frames_out, "loss_figure": fig}


def stop_training(name):
    """Stop a run; it saves its progress. train() with the same name and start=True continues it."""
    name = _name(name)
    rd = _run_dir(name)
    info = json.load(open(os.path.join(rd, "run.json")))
    open(os.path.join(rd, "STOP"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if _alive(info.get("pid")):
        os.killpg(os.getpgid(info["pid"]), signal.SIGTERM)
        for _ in range(60):
            if not _alive(info["pid"]):
                break
            time.sleep(1)
    print("Run `%s` stopped; its progress is saved and it can be continued." % name)
    return training_status(name, plot=False)


# ----------------------------------------------------------------------------- storage
# What the project keeps (persistent, counted against the storage quota) and what sits in the
# temporary folder. Each entry: (kind, label, root folder, kept).
def _areas():
    return [("model", "trained model", _models_dir(), True),
            ("prediction", "prediction", os.path.join(_workspace(), "predictions"), True),
            ("pretrained", "downloaded pretrained model", _zoo_dir(), False),
            ("run", "training run", os.path.join(_scratch(), "runs"), False),
            ("training_data", "training data", os.path.join(_scratch(), "training_data"), False)]


def _du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                n += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return n


def _size(n):
    return "%.1f GB" % (n / 1e9) if n >= 1e9 else "%.0f MB" % (n / 1e6) if n >= 1e6 else "%.0f KB" % (n / 1e3)


def _items():
    out = []
    for kind, label, root, kept in _areas():
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(d):
                out.append({"kind": kind, "label": label, "name": os.path.basename(d), "path": d, "kept": kept,
                            "bytes": _du(d), "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(d)))})
    return out


def _saved_line(path):
    """One line after something was written to the project folder: its size and the folder's total."""
    print("\nSaved to persistent storage: %s; this project's CDeep3M folder now holds %s (%.1f GB free). "
          "To see or free space: ask what CDeep3M is storing." % (_size(_du(path)), _size(_du(_workspace())),
                                                                 _free_gb(_workspace())))


def storage():
    """What CDeep3M keeps for this project and what is in the temporary folder, with sizes."""
    items = _items()
    ws = _workspace()
    print("## CDeep3M storage\n")
    print("**Kept** in `%s` (persistent storage: survives pod restarts, counts against the quota): **%s**, "
          "%.1f GB free.\n" % (ws, _size(_du(ws)), _free_gb(ws)))
    kept = sorted([i for i in items if i["kept"]], key=lambda i: -i["bytes"])
    if kept:
        print("| What | Name | Size | Last changed |\n|---|---|---|---|")
        for i in kept:
            print("| %s | %s | %s | %s |" % (i["label"], i["name"], _size(i["bytes"]), i["date"]))
    else:
        print("Nothing yet: no trained models or predictions.")
    sc = _scratch()
    tmp = sorted([i for i in items if not i["kept"]], key=lambda i: -i["bytes"])
    print("\n**Temporary** in `%s` (lost when the pod stops; pretrained models download again on use): "
          "**%s**, %.1f GB free.\n" % (sc, _size(_du(sc)), _free_gb(sc)))
    if tmp:
        print("| What | Name | Size | Last changed |\n|---|---|---|---|")
        for i in tmp:
            print("| %s | %s | %s | %s |" % (i["label"], i["name"], _size(i["bytes"]), i["date"]))
    else:
        print("Empty.")
    print("\nAnything listed can be deleted by name; nothing is deleted without the user's confirmation.")
    return {"kept_bytes": _du(ws), "temporary_bytes": _du(sc), "items": items}


def delete(names, kind=None, confirm=False):
    """Delete trained models, predictions, downloaded pretrained models, training runs or training data
    by name. Without confirm=True it only shows what would be deleted and its size."""
    names = [_name(n) for n in (names if isinstance(names, (list, tuple)) else [names])]
    items = _items()
    found, problems = [], []
    for n in names:
        hit = [i for i in items if i["name"] == str(n) and (kind is None or i["kind"] == kind)]
        if not hit:
            problems.append("`%s`: not found%s (the storage report lists what there is)" % (n, " as a %s" % kind if kind else ""))
        elif len(hit) > 1:
            problems.append("`%s` is both %s; say which (kind=%s)" % (
                n, " and ".join("a " + h["label"] for h in hit), " or ".join('"%s"' % h["kind"] for h in hit)))
        else:
            found.append(hit[0])
    for i in found:
        if i["kind"] == "run":
            try:
                pid = json.load(open(os.path.join(i["path"], "run.json"))).get("pid")
            except (OSError, ValueError):
                pid = None
            if _alive(pid):
                problems.append("`%s` is still training; stop it first" % i["name"])
                found.remove(i)
    print("## Delete\n")
    if found:
        print("| What | Name | Size | Where |\n|---|---|---|---|")
        for i in found:
            print("| %s | %s | %s | %s |" % (i["label"], i["name"], _size(i["bytes"]),
                                           "persistent storage" if i["kept"] else "temporary folder"))
    for p in problems:
        print("\n- " + p)
    total = sum(i["bytes"] for i in found)
    if found and not confirm:
        print("\nNothing has been deleted. This frees %s; it happens only when the user confirms." % _size(total))
        return {"deleted": [], "would_delete": [i["name"] for i in found], "bytes": total}
    done = []
    for i in found:
        roots = [os.path.realpath(r) for _, _, r, _ in _areas()]
        real = os.path.realpath(i["path"])
        if os.path.dirname(real) not in roots:          # only a direct entry of a CDeep3M folder
            print("\n- `%s`: not inside a CDeep3M folder, left alone" % i["name"])
            continue
        shutil.rmtree(real)
        done.append(i["name"])
    if done:
        print("\nDeleted %d item%s, %s freed. Persistent storage now holds %s (%.1f GB free)." % (
            len(done), "" if len(done) == 1 else "s", _size(sum(i["bytes"] for i in found if i["name"] in done)),
            _size(_du(_workspace())), _free_gb(_workspace())))
    return {"deleted": done, "bytes": sum(i["bytes"] for i in found if i["name"] in done)}


def finish_training(name, model_name=None):
    """Turn a training run into a model for predict(): export, re-estimate BatchNorm, convert to ncnn."""
    name = _name(name)
    rd = _run_dir(name)
    info = json.load(open(os.path.join(rd, "run.json")))
    if _alive(info.get("pid")):
        raise RuntimeError("run %s is still training; wait for it or stop_training(%r) first" % (name, name))
    model_name = model_name or name
    if model_name in ZOO:
        raise ValueError("choose a model name that is not a pretrained model's")
    md = os.path.join(_models_dir(), model_name)
    log = os.path.join(rd, "finish.log")
    print("## New model: %s\n" % model_name)
    print("| Network | Updates trained | Exported |\n|---|---|---|")
    done = []
    for f in info["settings"]["frames"]:
        ck = os.path.join(rd, f, "latest.pt")
        if not os.path.isfile(ck):
            print("| %s | 0 | no checkpoint, skipped |" % f)
            continue
        base = os.path.join(_model_path(info["base_model"]), f)
        if not os.path.isdir(base):
            get_model(info["base_model"], frames=f)
        fd = os.path.join(md, f)
        os.makedirs(os.path.join(fd, "trainedmodel"), exist_ok=True)
        shutil.copy(os.path.join(base, "deploy.prototxt"), fd)
        out = os.path.join(fd, "trainedmodel", "%s_trained.caffemodel" % f)
        rc = _run(["python3", "-m", "trainer.export_caffemodel", "--deploy", os.path.join(base, "deploy.prototxt"),
                   "--checkpoint", ck, "--template", _caffemodel(base), "--out", out, "--data", info["data"],
                   "--threads", info["settings"]["threads"]], log, cwd=ANYWHERE, env=dict(os.environ, PYTHONPATH=ANYWHERE))
        if rc != 0:
            raise RuntimeError("export failed:\n" + _tail(log))
        shutil.rmtree(os.path.join(fd, "ncnn"), ignore_errors=True)
        if _run([ANYWHERE + "/ncnn/convert_model.sh", os.path.join(fd, "deploy.prototxt"), out,
                 os.path.join(fd, "ncnn")], log) != 0:
            raise RuntimeError("conversion failed:\n" + _tail(log))
        upd = int((re.findall(r"update (\d+)\)", _tail(log, 30)) or ["0"])[-1])
        print("| %s | %d | yes |" % (f, upd))
        done.append(f)
    json.dump({"base_model": "none (from scratch)" if info["from_scratch"] else info["base_model"],
               "run": name, "updates": info["settings"]["updates"], "settings": info["settings"],
               "made": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(md, "source.json"), "w"), indent=1)
    print("\nNew model `%s` (networks: %s) saved in `%s`; it can now segment images." % (model_name, ", ".join(done), md))
    _saved_line(md)
    return {"model": model_name, "folder": md, "frames": done}

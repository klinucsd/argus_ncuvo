"""CDeep3M segmentation of microscopy images, for agents.

One function per task. Each prints a short Markdown report that starts with the settings it used,
so an answer can show exactly how a result was produced, and returns a dict for further work.

    import cdeep3m_lib as cd
    cd.environment()                                   # hardware, folders, time estimates
    cd.settings()                                      # every setting, its value and meaning
    cd.set_defaults(updates=5000)                      # change a default for this project
    cd.models()                                        # pretrained and own models
    cd.predict("images/", model="sbem-mitochondria")   # segment
    cd.prepare_training("train/images", "train/labels", name="mito")
    cd.train("sbem-mitochondria", "mito", name="mito-ft")             # shows the plan only
    cd.train("sbem-mitochondria", "mito", name="mito-ft", start=True) # starts in the background
    cd.training_status("mito-ft"); cd.stop_training("mito-ft")
    cd.finish_training("mito-ft")                      # -> a model usable by predict()
    cd.evaluate("<prediction name>", "labels/"); cd.show("<prediction name>", labels="labels/")

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
import tarfile
import time
import urllib.request

import numpy as np

LIB_VERSION = 4
CDEEP3M2 = os.environ.get("CDEEP3M2", "/opt/cdeep3m2")
ANYWHERE = os.environ.get("CDEEP3M_ANYWHERE", "/opt/cdeep3m-anywhere")
FRAMES = ("1fm", "3fm", "5fm")
IMAGE_EXT = (".png", ".tif", ".tiff")

# Pretrained models from the CDeep3M model zoo (https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models).
ZOO = {
    "sbem-mitochondria": {
        "title": "SBEM and ssTEM mitochondria",
        "about": "Mitochondria in serial block-face SEM and serial-section TEM images",
        "url": "https://cildata.crbs.ucsd.edu/media/model_display/50681/SBEM_Mitochondria.tar",
        "doi": "10.7295/W9CDEEP3M50681",
        "size": "653 MB",
    },
}

# Every setting: default and meaning. Functions take any of them as keyword arguments;
# set_defaults() changes them for the whole project.
SETTINGS = {
    "frames": ("1fm", "Networks to use: 1fm (one slice), 3fm, 5fm (3 or 5 neighbouring slices), or a "
                      "combination such as 1fm,3fm,5fm, whose predictions are averaged (more accurate, slower)"),
    "augspeed": (1, "Prediction: how many flipped and rotated copies of each image are predicted and averaged. "
                    "1 = all (best quality), 2, 4, 10 = one copy (fastest)"),
    "device": ("auto", "auto (the GPU when there is one), gpu or cpu"),
    "precision": ("auto", "Prediction on the CPU: auto (int8 when the model has an int8 version), fp32 or int8"),
    "threshold": (0.5, "Probability above which a pixel counts as the object, for masks and scores"),
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
    print("Change one for a single run by passing it to the function, e.g. `train(..., updates=5000)`; "
          "change it for the whole project with `set_defaults(updates=5000)`.\n")
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
def models():
    """List pretrained models and the project's own trained models."""
    print("## Models\n")
    print("| Model | What it segments | Source | Available here |\n|---|---|---|---|")
    for k, z in ZOO.items():
        here = os.path.isdir(os.path.join(_zoo_dir(), k, "1fm"))
        print("| %s | %s | model zoo, DOI %s (%s) | %s |" % (k, z["about"], z["doi"], z["size"],
                                                           "yes" if here else "downloaded on first use"))
    own = []
    for d in sorted(glob.glob(os.path.join(_models_dir(), "*"))):
        name = os.path.basename(d)
        if name in ZOO:
            continue
        info = {}
        try:
            info = json.load(open(os.path.join(d, "source.json")))
        except (OSError, ValueError):
            pass
        frames = [f for f in FRAMES if os.path.isdir(os.path.join(d, f))]
        own.append(name)
        print("| %s | trained here from %s (%s updates) | `%s` | yes (%s) |" % (
            name, info.get("base_model", "?"), info.get("updates", "?"), d, ", ".join(frames)))
    return {"pretrained": list(ZOO), "own": own}


def _caffemodel(frame_dir):
    cands = glob.glob(os.path.join(frame_dir, "trainedmodel", "*.caffemodel"))
    if not cands:
        raise FileNotFoundError("no .caffemodel in %s/trainedmodel" % frame_dir)
    num = lambda p: int((re.findall(r"(\d+)", os.path.basename(p)) or ["0"])[-1])
    return sorted(cands, key=num)[-1]


def get_model(name, frames=None, calibration_images=None):
    """Make a model ready for prediction: download a pretrained one if needed, convert to ncnn.
    calibration_images: a folder of training images to also build an int8 version (faster on the CPU)."""
    d = _model_path(name)
    fl = _frames(frames or "1fm")
    log = os.path.join(_scratch(), "get_model.log")
    if not os.path.isfile(os.path.join(d, "1fm", "deploy.prototxt")):
        if name not in ZOO:
            raise FileNotFoundError("unknown model %r; pretrained models: %s" % (name, ", ".join(ZOO)))
        z = ZOO[name]
        tmp = os.path.join(_scratch(), "download_" + name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        print("Downloading %s (%s) from the CDeep3M model zoo ..." % (z["title"], z["size"]), flush=True)
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
        json.dump({"source": "model zoo", "title": z["title"], "doi": z["doi"], "url": z["url"]},
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
    model = _name(model)
    s = _effective(overrides)
    files = _check_folder(images, "prediction")
    name = name or "%s_%s" % (model, _stamp())
    gpu = _gpu()
    print("## Prediction: %s\n" % name)
    print("**Settings used**\n")
    print(_settings_table(dict(s, model=model, images=images), ["model", "images", "frames", "augspeed",
                                                                 "device", "precision", "threshold"]))
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
    meta = {"name": name, "model": model, "images": os.path.abspath(images), "image_files": [os.path.basename(f) for f in files],
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
    print("\nPrediction name for evaluate() and show(): `%s`" % name)
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
    print("\nTemporary: kept only while the pod runs. Training data name for train(): `%s`" % name)
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
    model = _name(model)
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
    print(_settings_table(dict(s, model="none (from scratch)" if from_scratch else model, data=training_data),
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
        print("\nThis run is already training (process %s). Use training_status(%r)." % (info["pid"], name))
        return {"name": name, "running": True}
    if not start:
        print("\nNothing started yet. To start, call `train(..., start=True)` with the same arguments.")
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
    print("\nStarted in the background (process %d). It keeps running between questions; "
          "ask for training_status(%r) at any time." % (p.pid, name))
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
    print("Run %s stopped; progress is saved. Continue it with train(..., name=%r, start=True)." % (name, name))
    return training_status(name, plot=False)


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
    print("\nModel saved in `%s`. Use it with predict(images, model=%r, frames=%r)." % (md, model_name, ",".join(done)))
    return {"model": model_name, "folder": md, "frames": done}

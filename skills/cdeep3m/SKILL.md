---
name: cdeep3m
description: Segment electron, light or X-ray microscopy images with CDeep3M deep learning — mitochondria, membranes, nuclei, vesicles. Predict with pretrained or own models, fine-tune or train on labelled slices (GPU or CPU, in the background), track training, evaluate against expert labels (Dice, IoU), show segmentations. Settings such as updates, learning rate, networks and speed can be changed per run or per project.
---

# CDeep3M segmentation

Everything is done by the functions of `cdeep3m_lib`, shipped with this skill. Call them; do not
call CDeep3M's shell scripts or write your own segmentation, training or scoring code. Each function
prints a Markdown report that begins with the settings it used: present that report as printed, under
its own heading (do not add another heading above it), with at most one sentence of your own around it.

```python
import importlib, cdeep3m_lib as cd
importlib.reload(cd)          # pick up the current version
```

This file and the table below are the whole interface: call the functions directly, without opening
`cdeep3m_lib.py` first. Each function's arguments are its settings (`cd.settings()` lists them).

**Do only what the request asks, in one script.** Pick the functions from the table that answer it
and stop there. A question about what is possible, what is available or how long something would take
is answered with reports (`environment`, `models`, `settings`, `sample_data`): nothing is downloaded,
prepared, trained or predicted for it. Downloads, data preparation, prediction and training happen
only when the user asks for that work.

Use the user's words for files and folders; when they give none, the sample data
(`cd.sample_data()`) is there for trying things out.

## Which function for which request

| The user wants to … | Call |
|---|---|
| know what hardware, folders and speed to expect | `cd.environment()` |
| see or change settings | `cd.settings()`; for this request only pass them as arguments; for the whole project `cd.set_defaults(name=value, ...)` |
| know which models exist | `cd.models()` |
| try the sample data | `cd.sample_data()` |
| segment images | `cd.predict(images, model=..., name=...)` |
| score a segmentation against expert labels | `cd.evaluate(prediction_name, labels)` |
| see a segmentation | `cd.show(prediction_name, labels=...)` → embed the returned PNG with `![...](path)` |
| prepare labelled slices for training | `cd.prepare_training(images, labels, name=...)` |
| train or fine-tune | `cd.train(model, data_name, name=...)` first (plan only), then with `start=True` |
| follow training | `cd.training_status(run_name)` → embed the loss curve |
| stop / continue training | `cd.stop_training(run_name)` / `cd.train(..., name=run_name, start=True)` |
| use the trained network | `cd.finish_training(run_name, model_name=...)`, then `cd.predict(..., model=model_name)` |

Wherever a function takes a prediction, training data, run or model, pass either its name or what the
earlier function returned. `model=None` in `train` trains from scratch (learning rate 0.01 unless the user sets one);
otherwise it fine-tunes the given model, the usual and much faster way.

## Rules

1. **Training needs the user's go-ahead.** Call `train(...)` without `start=True` first, show the plan
   (settings and estimated time) and ask whether to start. Start only after the user agrees, in a
   later request, with exactly the settings shown. The same for predicting more than a few hundred
   images: say how long it will take before running it.
2. **Training runs in the background.** `train(..., start=True)` returns at once. Do not wait or poll
   in a loop; report `training_status` when the user asks.
3. **Settings are named, not invented.** Use only the settings `cd.settings()` lists. When the user
   asks to change something, pass the matching setting; if nothing matches, say so.
4. **Pairing for scores:** `evaluate` and `show` pair predictions and labels in sorted file order.
   Make sure the label folder holds the labels of the same images, in the same order.
5. Pretrained models, training data and training runs live in the temporary folder `/tmp/cdeep3m`
   (large; lost when the pod stops). Own trained models (from `finish_training`), predictions and
   settings are kept in the project folder `cdeep3m/` next to the notebook. When a pod restart has
   cleared a run, prepare the data and start it again.
   Embed figures with the path the function prints.

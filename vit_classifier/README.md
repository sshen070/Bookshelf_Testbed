# ViT ROI Classifier

WAFT answers **where** something moved. This answers **what is there**.

Standalone: it imports nothing from the deprecated `src/` pipeline at the repo
root, and is not wired into it. It consumes WAFT's on-disk output — event
bursts from `camera_event_detector.py` and box JSON from `flow_boxes_demo.py` —
and is meant to run *after* de tection, never inside WAFT's per-frame hot loop.

```
WAFT (motion detection)                    this package (classification)
─────────────────────────                  ────────────────────────────────
camera_event_detector.py  ──> event dir ──> classify-event ──> what's in the box
                              (live)   └──> watch           and what changed
flow_boxes_demo.py        ──> boxes.json ─> classify-image
```

## Why a frozen ViT + k-NN, not a fine-tuned classifier

The object vocabulary is small, known, and changes whenever someone puts a new
object in the scene. A fine-tuned head would mean a training run per vocabulary
change plus a labeled dataset that doesn't exist. Here the ViT is **never
trained** — it's a frozen DINOv2 feature extractor, and classification is a k-NN
vote over a bank of exemplar crops. Adding a class means dropping 5–10 crops
into a new folder and re-running `build-bank`, which takes seconds. The trade is
a few points of accuracy for something maintainable by whoever is running the
rig.

## The two-frame trick

A motion box brackets the region that *changed between two frames*. Classify
only the later frame and, for an object that left the scene, you get "empty
background" — true and useless. Cropping the **same box from both frames** and
diffing the two predictions recovers the actual event:

| before → after | reported |
|---|---|
| `mug` → `background` | `removed` |
| `background` → `bottle` | `placed` |
| `mug` → `bottle` | `substituted` |
| `mug` → `mug` | `moved_or_same_class` |
| either side below `min_similarity` | `uncertain` |

`moved_or_same_class` is an honest limit, not a bug: class labels alone cannot
separate "the same object shifted" from "swapped for another of its kind."
`uncertain` means nothing in the bank was close enough — usually a genuinely
new object, which is a signal to grow the bank rather than something to force
into the nearest existing class.

For an event burst the pair is the earliest `pre_*` frame and the latest
`post_*` frame — the widest interval available, so a slow placement has
actually finished by the "after" side.

## Setup

```bash
pip install -r vit_classifier/requirements.txt
```

If WAFT's own environment is already set up it **already satisfies everything**
— it pins `timm==1.0.27` alongside torch/torchvision, so just use that
interpreter. On a Jetson, install NVIDIA's CUDA aarch64 torch wheel first (see
WAFT's `README_EVENT_DETECTION.md`), then `pip install timm` on top; the generic
PyPI wheel is CPU-only or x86 and won't use the GPU.

## Usage

Bootstrap order matters — the bank has to exist before anything can be
classified.

```bash
# 1. dump crops from real WAFT events (every box, every frame of each burst)
python -m vit_classifier extract-crops \
    --events-dir WAFT/WAFT/outputs/camera_events/events

# 2. hand-sort exemplars/_unsorted/ into one folder per object:
#      vit_classifier/exemplars/mug/
#      vit_classifier/exemplars/bottle/
#      vit_classifier/exemplars/background/   <- empty-region crops, mostly the pre_* ones
#    (folders starting with _ are ignored, so _unsorted/ is safe to leave)

# 3. embed and cache
python -m vit_classifier build-bank

# 4. check the bank is actually separable before trusting it
python -m vit_classifier eval-bank

# 5. classify
python -m vit_classifier classify-event --event-dir WAFT/WAFT/outputs/camera_events/events/<event_name>
python -m vit_classifier classify-event --events-dir WAFT/WAFT/outputs/camera_events/events
python -m vit_classifier classify-image --image frame.png --boxes-json boxes.json --out result.json
```

The best exemplars are crops the detector actually produced — same camera,
scale, and lighting — which is why `extract-crops` pulls from all frames of
each burst rather than just the transition pair.

## Live mode

`watch` runs alongside `camera_event_detector.py` and classifies each burst as
it lands, instead of walking a directory of finished events after the fact.

```bash
# terminal 1: the detector, writing events as motion happens
./run_detection.sh

# terminal 2: the classifier, picking them up as they complete
python -m vit_classifier watch --events-dir WAFT/WAFT/outputs/camera_events/events
```

Each result is written to `<event_dir>/vit_result.json` — the same JSON
`classify-event --out` produces — next to WAFT's own `waft_results/`.

### When is an event "complete"?

An event directory appears on disk well before it is finished, so the watcher
gates on the lifecycle fields in `event.json`:

```
status       pending_post_frames ──> ready_for_waft
waft.status  not_started ──> running ──> complete | failed
```

**`status == "ready_for_waft"` is the signal that matters.** The detector sets
it only after the last post-frame has been copied in, so every frame the
manifest names is on disk by then. Classify earlier and the burst has no
post-frame yet — the before/after pair degrades to pre-vs-trigger, and you get a
verdict on a placement that is still mid-air.

`waft.status` is about **contention, not data**. Nothing here reads
`waft_results/`; frames and boxes both come from the detector and are final
before any review starts. But with `--auto-waft` on, WAFT's review subprocess
owns the GPU for ~20s right after the event lands, and on the Jetson's shared
8GB that is the wrong moment to also be loading a ViT. So:

| `--wait-for-waft` | behaviour |
|---|---|
| `auto` (default) | Classify once the burst is complete, unless a review is *currently running* — then wait for it. Nothing to wait for when `--auto-waft` is off. |
| `always` | Only after `waft.status == complete`. Use when you want the review's montage and the classification to line up 1:1. |
| `never` | Classify the moment the burst is written. Lowest latency, competes with the review for the GPU. |

`--waft-timeout` (default 180s) stops a review whose process died without
updating `event.json` from stranding the event forever.

### Restarts and backlogs

By default the watcher ignores everything already on disk — it is for what
happens next, and history is what `classify-event --events-dir` is for.
`--backfill` classifies pre-existing events too, skipping any that already have
a result file, which is exactly what makes a restart pick up what it missed
while it was down. `--reclassify` forces those to be redone.

Events with nothing classifiable in them (grid-triggered bursts carry no boxes)
get a `{"skipped": ...}` result file, so they aren't re-examined on every
restart.

Other things worth knowing:

- The model and bank load **once**, before the first event — an event lands about
  a second after its trigger, and paying a ten-second model load per event would
  put every result well behind the burst it describes. Restart the watcher after
  editing exemplars; the bank is not re-read while it runs.
- Reads are forgiving by design. Both WAFT processes rewrite `event.json` in
  place, so a poll can land mid-write; a half-written manifest means "not yet",
  and the next poll is a second away.
- Polling, not inotify: no extra dependency, and it behaves the same over a
  network mount, where inotify silently reports nothing.
- A burst that fails to classify is logged and skipped, not fatal — the watcher
  is meant to survive a whole session.

### Checking the bank

`eval-bank` classifies each exemplar against a bank with that exemplar
removed, and prints mean cosine similarity within and between every class.
Run it after every rebuild: adding exemplars can *hurt*, and a k-NN gives you
no training curve to notice that from.

Read the similarity matrix, not just the accuracy. A class is separable when
its diagonal clearly exceeds its own row, and two classes bleeding together
show up as a hot off-diagonal cell long before they cost enough accuracy to
notice. Both numbers are optimistic on a small bank of similar photos, and
neither measures the gap that actually bites — exemplars shot on a phone
versus crops the detector produced. A high score means "nothing is
fundamentally broken", not field accuracy.

### As a library

```python
from vit_classifier import load_event, scale_boxes
from vit_classifier.classifier import RoiClassifier

event = load_event("WAFT/WAFT/outputs/camera_events/events/event_.../")
before, after = event.pair_for_transition()

clf = RoiClassifier({"img_size": 224, "device": "cuda"})
clf.ensure_bank("vit_classifier/exemplars", "vit_classifier/.cache/exemplar_bank.npz")

for box, t in zip(event.boxes, clf.classify_pair(before.load(), after.load(), event.boxes)):
    print(box.to_list(), t.before.label, "->", t.after.label, "=", t.event)
```

`boxes`, `waft_events`, and `config` import cleanly **without torch or timm** —
only `classifier` needs the model stack. Box parsing and crop extraction work
on a bare machine.

## The scaling trap

WAFT reports boxes in the resolution the **flow** was computed at, not the
resolution the frames were **saved** at. `flow_boxes_demo.py --width 320` writes
boxes in 320×240 while the PNGs may be 960×540. Cropping a 320×240 box out of a
960×540 image silently addresses the wrong third of the frame, and the
classifier confidently labels the wrong object.

`event.json` records what it used as `boxes.processing_size`, and the CLI
rescales automatically (and says so when it does). `boxes.json` from
`flow_boxes_demo.py` records **no** size, so a library caller must know the flow
resolution and route through `scale_boxes` themselves. `crop()` cannot detect
the mismatch for you.

Two related gotchas handled here: `event.json` frame paths are relative to the
WAFT repo root and break the moment an event folder is copied, so frames are
resolved by basename against `<event_dir>/frames/`; and `frames_annotated/` is
never read, since those PNGs have boxes painted on them.

## Configuration

Everything lives in [`config.yaml`](config.yaml). Relative paths resolve against
this package, not the working directory, so the CLI behaves the same from any
cwd. The knobs that matter:

| Key | Why you'd change it |
|---|---|
| `model.name` | `vit_base_patch14_dinov2.lvd142m` is stronger and ~4× slower. |
| `model.img_size` | `224` interpolates DINOv2's native 518px down for CPU speed. Set `null` for full accuracy on a GPU. |
| `bank.knn_k` | Keep at or below your smallest per-class exemplar count, or that class can never win a full vote (`build-bank` warns). |
| `bank.min_similarity` | How readily an unseen object gets forced into a known class. Raise it when the bank is small. |
| `crops.pad_frac` | Context around each box. Motion boxes hug the moving pixels tightly. |
| `transitions.background_class` | Must match a real folder under `exemplar_dir`, or removals can never be reported. |

The bank cache records the model name, input size, and a fingerprint of the
exemplar files — edit or add an exemplar and it rebuilds automatically.

## Cost

Two ViT forward passes per box for `classify_pair`. This is **not** designed for
the always-on path: run it on saved events after the fact, the same way WAFT's
own `--auto-waft` confirmation is rate-limited rather than per-frame.

## Tests

```bash
python -m pytest vit_classifier/tests -q          # 84 tests, no torch needed
VIT_TEST_MODEL=1 python -m pytest vit_classifier/tests -q   # +6 model-backed
```

The model-backed tests download ~85MB of pretrained weights on first run, so
they're opt-in rather than slowing every run and breaking offline ones.

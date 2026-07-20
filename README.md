# Bookshelf Testbed — Change-Detection Pipeline

A physical-shelf lab proxy for dam crack detection. A classical, fixed-reference
computer-vision pipeline (frame registration + multi-scale SSIM) auto-labels
"changed" vs "normal" regions in shelf photos with no manual bounding-box
annotation required; those pseudo-labels train a YOLOv8 "change" detector,
which is then cross-validated against the classical pipeline to check it
learned genuine structural change rather than noise. A Raspberry Pi + camera
can stream a live feed for real-time monitoring.

The design choices here (fixed-reference monitoring rather than
frame-to-frame diffing, explicit lighting-drift robustness, multi-scale
detection of both abrupt and subtle changes) are meant to transfer to real
outdoor dam imagery, where cracks are smaller, lower-contrast, and develop
over much longer timescales than anything staged in the lab.

## How it works, end to end

```
data/reference/*.jpg  (fixed baseline, varied lighting, no changes)
        │
        ▼
data/raw/<session>/*.jpg  (new captures, one folder per distinct event)
        │
        ▼  python run_pipeline.py autolabel
   registration (align to best-matching reference)
   → lighting normalization (CLAHE + gray-world)
   → multi-scale SSIM dissimilarity map
   → connected components → bounding boxes
        │
        ▼
data/processed/pseudo_labels/  (YOLO-format labels, no human ever drew a box)
data/processed/events.jsonl    (per-image verdict: change / no_change / registration_failed)
        │
        ▼  python run_pipeline.py build-dataset
data/yolo_dataset/  (train/val/test split, grouped by session)
        │
        ▼  python run_pipeline.py train
runs/yolo/change_detector*/weights/best.pt
        │
        ▼  python run_pipeline.py crossvalidate  /  live
Compare YOLO's detections against the classical pipeline's on held-out or
live images — persistent disagreement is the diagnostic signal.
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If your camera exports HEIC (iPhone), convert before dropping images in:
```
python convert_heic.py data          # recursively converts *.heic/*.heif to *.jpg, deletes originals
python convert_heic.py data --keep-original
```

## Capturing data

**`data/reference/`** — the fixed baseline. Same static camera position every
time (registration corrects small drift, not a moved or re-angled camera).
No perturbations. Capture several shots across different lighting conditions
(morning/evening/lamp on/overcast) — this bank is what the pipeline matches
incoming frames against instead of fighting lighting drift with one photo.

**`data/raw/<session_name>/`** — everything else, normal or changed. **One
directory per distinct physical event** (a specific book toppled, a specific
binder shift, a lighting-only variant with nothing touched) — never lump
multiple different arrangements into one folder, and never split one
arrangement across several folders. Splitting for training is grouped by
session specifically so near-duplicate frames from the same event can't leak
between train and val — a folder holding more than one real event defeats
that. The folder name itself doesn't matter, just its uniqueness per event.

No manual labeling anywhere — `autolabel` generates the pseudo-labels for
both reference and raw images automatically.

## Pipeline stages (`run_pipeline.py`)

Single entry point; `python run_pipeline.py <stage> [options]`.

| Stage | What it does |
|---|---|
| `autolabel` | Registers + diffs every image under `data/raw/` against the reference bank, writes YOLO-format pseudo-labels to `data/processed/pseudo_labels/`, and logs one verdict per image to `data/processed/events.jsonl`. Images that fail registration are excluded from the pool, not silently mislabeled. |
| `build-dataset` | Splits the pseudo-labeled pool into train/val/test (grouped by session, see `dataset_split` in `pipeline.yaml`), writes `data/yolo_dataset/` and `data.yaml`. Fails loudly with a clear message (rather than a deep Ultralytics stack trace) if too few sessions leave train or val empty. |
| `train` | Trains YOLOv8 on `data/yolo_dataset/` using `configs/yolo.yaml`. Ultralytics auto-increments the run folder (`change_detector`, `change_detector-2`, ...) if a previous run already exists there. |
| `crossvalidate` | Runs both the classical pipeline and the trained YOLO model on a directory of images, logs per-image agreement to `data/processed/crossvalidation.jsonl`, and prints a summary — agreement rate, and counts of "only classical" / "only YOLO" disagreements (your debugging signal: only-classical usually means an under-represented change type in training; only-YOLO means check for hallucination vs. real generalization). |
| `live` | Connects to a Raspberry Pi's MJPEG stream (see below), runs both detectors on every frame in real time, shows an annotated window, and saves every changed frame into a new `data/raw/live_<timestamp>/` session — a monitoring run doubles as data collection for future retraining. |
| `add-reference` | The only sanctioned way to grow `data/reference/`. Manual and human-confirmed by design — see "Growing the reference bank" below. |
| `reset` | Wipes derived/regenerable artifacts to start clean. See "Resetting" below. |
| `all` | Runs `autolabel` → `build-dataset` → `train` in sequence. |

### Arguments

| Flag | Used by | Meaning |
|---|---|---|
| `--pipeline-config` | all stages | Path to the classical-pipeline config (default `configs/pipeline.yaml`). |
| `--yolo-config` | `train`, `all` | Path to the YOLO training config (default `configs/yolo.yaml`). |
| `--image-dir` | `crossvalidate` | **Required.** Directory of images to evaluate, e.g. a held-out session. |
| `--weights` | `crossvalidate`, `live` | **Required.** Path to trained weights, e.g. `runs/yolo/change_detector-2/weights/best.pt`. |
| `--conf` | `crossvalidate`, `live` | YOLO confidence threshold (default `0.25`). |
| `--pi-host` | `live` | Raspberry Pi's IP address or hostname. |
| `--pi-port` | `live` | Port `stream_server.py` is serving on (default `8000`). |
| `--headless` | `live` | Skip the `cv2.imshow` window — use when running over SSH with no display. |
| `image` (positional) | `add-reference` | **Required.** Path to the candidate normal-state image to add. |
| `--yes` | `add-reference`, `reset` | `add-reference`: skip the interactive y/n confirmation. `reset`: required to actually wipe `--raw`/`--runs`. |
| `--no-preview` | `add-reference` | Skip the side-by-side `cv2.imshow` preview (needed when running headless). |
| `--raw` | `reset` | Also wipe `data/raw/` captures. Irreversible — requires `--yes`. |
| `--runs` | `reset` | Also wipe `runs/yolo/` trained model weights. Irreversible — requires `--yes`. |
| `--dry-run` | `reset` | Print what would be deleted without deleting anything. |

### Examples

```
python run_pipeline.py autolabel
python run_pipeline.py build-dataset
python run_pipeline.py train
python run_pipeline.py crossvalidate --image-dir data/raw/holdout_session --weights runs/yolo/change_detector-2/weights/best.pt
python run_pipeline.py live --pi-host 192.168.1.42 --weights runs/yolo/change_detector-2/weights/best.pt --headless
python run_pipeline.py add-reference data/raw/some_session/frame_003.jpg
python run_pipeline.py reset --dry-run
python run_pipeline.py all
```

Note: always invoke either `run_pipeline.py` or a module via `python -m
src.<module>` (e.g. `python -m src.autolabel`) from the repo root — running a
file directly as `python src/autolabel.py` breaks its `from src....` absolute
imports.

## Growing the reference bank (instead of resetting the baseline)

If you place an object back after a staged perturbation, it's rarely in the
*exact* original spot — a few mm of jitter is normal. Resetting the baseline
before every session would "fix" that, but it destroys the entire point of
fixed-reference monitoring: a slow, cumulative drift (the dam-crack analog)
would never accumulate into a detectable signal if the reference keeps
absorbing it as "the new normal."

Instead, `select_best_reference` (in `src/lighting.py`) does two-stage
matching: a cheap lighting-histogram prefilter shortlists candidates, then
the top few are actually registered and diffed, and whichever gives the
*lowest* residual difference is used. If the bank already contains a
confirmed-normal snapshot with similar jitter, it gets preferred over the
founding baseline — without ever discarding that founding baseline.

To add such a snapshot:
```
python run_pipeline.py add-reference data/raw/some_session/frame_003.jpg
```
This shows the closest existing reference side-by-side with your candidate,
reports the classical pipeline's own residual-diff verdict (it'll warn you
if the image still looks like a real change), and only copies it into
`data/reference/` on your explicit `y` confirmation — additive only, never
overwrites or deletes existing references. This step is deliberately manual:
an auto-updating baseline would let a real anomaly quietly become the new
"normal" the next time something drifts near it.

## Resetting for a new round of sessions

```
python run_pipeline.py reset --dry-run   # preview first
python run_pipeline.py reset             # clear derived artifacts only
```
By default this clears everything `autolabel`/`build-dataset`/`train`
regenerate — aligned frames, diff maps, the pseudo-label pool, the
`yolo_dataset` split, and the event/crossvalidation logs — and leaves
`data/raw/`, `data/reference/`, and `runs/` untouched, since those aren't
reproducible from anything else. Pass `--raw` and/or `--runs` (each requires
`--yes`) to additionally wipe raw captures or trained model weights.

## Live inference from a Raspberry Pi camera

Architecture: the Pi only captures and streams; all inference (classical +
YOLO) runs on your PC.

**On the Pi** — `pi/stream_server.py` serves an MJPEG HTTP stream using
`picamera2` (official Camera Module) or, with `--usb-cam`, plain OpenCV
capture for a USB webcam. No torch/ultralytics needed on the Pi.
```
sudo apt install -y python3-picamera2 python3-opencv
python3 pi/stream_server.py --port 8000
# or, USB webcam:
pip install -r pi/requirements-pi.txt
python3 pi/stream_server.py --usb-cam --port 8000
```
`stream_server.py` flags: `--port` (default 8000), `--width`/`--height`
(default 1280x960), `--fps` (default 15), `--usb-cam` (force OpenCV capture),
`--cam-index` (OpenCV camera index, only with `--usb-cam`).

**On the PC** — `python run_pipeline.py live --pi-host <pi-ip> --weights <path>`
connects to `http://<pi-host>:<pi-port>/stream.mjpg`, auto-reconnects on a
dropped stream, and annotates classical-pipeline boxes (orange) and YOLO
boxes (green) live. A new `data/raw/live_<timestamp>/` session starts
whenever a change is detected after 30s of quiet (tunable via
`--new-session-gap` when calling `src/live_infer.py` directly), so distinct
real events captured during one monitoring run still get separated properly
for later training splits.

### On-device motion detection (instead of dumb streaming)

`pi/motion_boxes.py` is an alternative to `stream_server.py` — run one or the
other, not both, since they both need to own the camera and most backends
(picamera2, a USB webcam) only allow one open handle at a time. Where
`stream_server.py` just serves raw frames and leaves all detection to the PC,
`motion_boxes.py` runs classical Farneback optical flow *on the device
itself*, frame to frame, live — no torch/WAFT/ultralytics involved, which is
what makes continuous on-device operation feasible on a Pi's CPU or a Jetson
Nano. It extracts real bounding boxes (threshold → morphology → connected
components → solidity filter → merge, the same pipeline `src/ssim_diff.py`
uses for SSIM and `WAFT/WAFT/flow_boxes_demo.py` uses for WAFT's flow —
duplicated with only cv2+numpy here, deliberately, so this file has zero
dependency on the rest of the project) at a small internal resolution
(default 160×120) and scales the results back up to the full capture
resolution.

```
python3 pi/motion_boxes.py --port 8000
python3 pi/motion_boxes.py --usb-cam --port 8000
```
Serves `http://<device-ip>:8000/stream.mjpg` (annotated live view) and
`/boxes.json` (latest detection: box list + timestamp), and saves
motion-triggered frames locally to `events/<session>/` — a new session
starts after `--new-session-gap` (default 30s) of no motion, matching the
same event-separation convention `live_infer.py` uses. That lets the device
run autonomously with no PC connected; copy/rsync `events/` into this
project's `data/raw/` later and run `autolabel` as usual.

**What this measures is different from the reference-bank pipeline**: Farneback
detects *motion between two consecutive live frames* (something actively
moving right now), not a settled state difference against a fixed reference
captured possibly weeks earlier under different lighting. Treat its boxes as
a live event trigger, not as another source of "change" pseudo-labels —
same caveat as WAFT's boxes, see `src/waft_bridge.py`.

## WAFT integration (optional, PC-side)

[WAFT](https://github.com/princeton-vl/WAFT) (vendored under `WAFT/`) is a
state-of-the-art dense optical flow model — given two frames, a per-pixel
motion field, not object detections. It needs its own environment
(torch/timm/xformers per `WAFT/WAFT/README.md`), separate from this
project's lightweight venv, and is too heavy to run live on a Pi/Jetson —
that role belongs to `pi/motion_boxes.py` above, which uses cheap classical
Farneback flow instead.

- `WAFT/WAFT/flow_boxes_demo.py` — runs inside WAFT's own environment.
  Computes flow for a sequence of frame pairs and extracts real bounding
  boxes from the flow-magnitude map (same threshold/morphology/connected-
  components/solidity/merge pipeline as everywhere else in this project),
  instead of the coarse fixed grid the repo's own `multi_frame_demo.py`
  reduces to.
- `src/waft_bridge.py` (`python run_pipeline.py waft-boxes --waft-python
  <path> --frames a.jpg b.jpg ...`) — invokes the above as a subprocess from
  this project and reads back the results.

Like Farneback, WAFT measures motion between two frames, not a state
difference against the reference bank — don't merge its output into
`build_dataset.py`'s "change" pseudo-label pool without deciding that's
actually what you want; see the docstrings in both files above for the
reasoning and a suggested use (feed a flagged event's settled frame into
`data/raw/<session>/` and let `autolabel.py` judge it against the reference
bank the normal way).

## Config reference

**`configs/pipeline.yaml`** — the classical pipeline. Key sections:
- `paths` — where everything reads from / writes to.
- `registration` — ECC (intensity-based, handles lighting-only drift) with
  ORB+RANSAC homography fallback (feature-based, handles a bumped camera).
  `max_reprojection_error_px` controls how much registration error is
  tolerated before a frame is flagged `registration_failed` instead of diffed
  against a misaligned reference.
- `lighting` — reference-bank selection (`reference_bank_metric`,
  `structural_shortlist_k`) and CLAHE/gray-world normalization strength.
- `ssim` — multi-scale window sizes/weights and `diss_threshold` (lower it
  for low-contrast/outdoor imagery; raise it if lab lighting noise causes
  false positives).
- `morphology` — noise cleanup and `min_component_area_px` before boxes are
  extracted.
- `bbox` — padding, merge IoU, and a sanity cap on boxes per image.
- `labeling` — pseudo-label class name and whether to keep label-free
  background images (recommended — teaches "lighting drift ≠ change").
- `dataset_split` — train/val/test ratios; `group_by_session` (keep this
  `true`) prevents near-duplicate frames from one event leaking across
  splits.

**`configs/yolo.yaml`** — YOLO training. Notably: starts from `yolov8n.pt`
(small model, less likely to memorize noisy pseudo-labels), freezes the
backbone for the first several epochs, and disables augmentations that don't
make sense for a fixed-viewpoint rig (`flipud`/`fliplr`/`mosaic` all `0`).

## Testing

```
python -m pytest tests/test_pipeline_smoke.py -v
```
Synthetic end-to-end smoke tests (no real shelf photos needed) covering
registration identity, SSIM diff on a no-change pair, detection of a
simulated toppled book, lighting-normalization suppressing a pure-brightness
diff, and the full `autolabel` path on a synthetic image.

## Interpreting cross-validation output

`crossvalidate` and `live` both report, per image:
- **Both flag change** / **both flag no-change** — agreement, good.
- **Only classical** — YOLO missed something the classical pipeline caught.
  Usually means that class of change is under-represented in the training
  pool; feed more examples of it back through `autolabel`.
- **Only YOLO** — YOLO flagged something classical didn't. Could be
  generalizing well (catching something the fixed SSIM threshold is too
  conservative for) or hallucinating — inspect the actual crop before
  trusting it either way.

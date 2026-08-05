# YOLO-World open-vocabulary detector

WAFT answers **where something moved**. `vit_classifier` answers **what is
there**, given a region. This answers **both at once, from text prompts** —
and exists to test whether that is a better trade than the two-stage pipeline.

Standalone: imports nothing from `WAFT/`, `vit_classifier/`, `video_link/`,
`device_status/`, or the deprecated `src/`. It reads WAFT's on-disk events and
`vit_classifier`'s result files, and writes into neither.

```
two-stage (today)                          this package
─────────────────────────────────          ────────────────────────────────
camera_event_detector.py                   detect ──> label + box in one pass
  Farneback motion ──> event burst           no trigger stage
        └──> vit_classifier ──> what           no exemplar bank
             (needs a ROI to classify)         no motion required
```

## Why this exists

Measured on this Orin Nano, sustained loops with `tegrastats` sampling `VDD_IN`:

| workload | fps | ms/frame | mW | mJ/frame | marginal mJ/frame |
|---|---|---|---|---|---|
| idle | — | — | 3197 | — | — |
| Farneback motion detection (CPU) | 4.00 | 251.8 | 5149 | 1287 | 488 |
| **YOLO-World 640px (GPU)** | **26–29** | **34–38** | 9861 | 376 | **253** |
| YOLOv8n 640px (GPU) | 38.12 | 25.8 | 7819 | 205 | 121 |

The always-on stage this would replace is **7× slower and uses 2× the energy
per frame** — and after paying it you still owe a classification pass
(264 ms for 14 ViT crops). The counterintuitive part is that the "cheap"
classical CV loses: Farneback is a serial CPU algorithm that pins cores for a
quarter second, while the GPU finishes a whole VLM detection in 34 ms.

Instantaneous draw is higher (9.9 W vs 5.1 W), so if you are power-capped
rather than latency-bound, duty-cycle it: at a matched 4 fps this costs
~1.0 W marginal against Farneback's ~2.0 W.

## What the swap actually changes

Run `compare` and this stops being abstract. On the one event in the tree:

```
  7 motion box(es) -> vit_classifier
  12 detection(s) -> YOLO-World
  1 region(s) both systems agree on
```

Every motion box brackets the **operator's hand** — that is what moved — and
`vit_classifier` correctly returns `uncertain` for most of them, because a
palm is not in its exemplar bank. Meanwhile YOLO-World finds the green boxes,
red boxes and books *that were never moving at all*.

That is the trade in one table:

| | two-stage | single-stage |
|---|---|---|
| triggers on | any motion, no vocabulary needed | only what you prompted for |
| finds static objects | **never** — no motion, no event | every frame |
| new class costs | 5 crops in a folder | typing a word |
| localizes | motion does; the ViT does not | yes |
| unknown object appears | archived as motion for review | **invisible** |

The last row is the real risk. Motion is a vocabulary-free proxy for
"something happened"; a prompt list is not. If coverage matters more than
efficiency, the cheap hybrid is this package plus WAFT's `change_detection`
(only 19 ms of that 252 ms) as a class-agnostic backstop.

## Accuracy against vit_classifier

Scored on 13 hand-labelled regions of one full-shelf frame, both systems in the
**same** five-class label space. `accuracy` gives each region the label of the
best-overlapping detection; a region no detection covers is a miss.

| class | YOLO-World | vit_classifier |
|---|---|---|
| books | 2/4 (2 missed) | **4/4** |
| containers | **5/5** | 1/5 |
| CSN_boxes | 1/2 | **2/2** |
| pickleballs | 1/1 | 1/1 |
| plushie | 0/1 (missed) | **1/1** |
| **total** | **9/13 (69%)** | **9/13 (69%)** |

Identical totals, near-opposite error profiles. The ViT reads `containers` as
`CSN_boxes` or `books` four times out of five — orange magazine files next to
orange plastic boxes and actual books is exactly the confusion a k-NN over 19
phone photos would make. YOLO-World gets all five, and misses things the ViT
never could.

Two caveats keep this honest. The ViT **cannot miss** — it is handed the box,
while YOLO-World has to find it, which is a harder task. And the ground truth
groups objects into regions ("a row of books") where YOLO-World detects
instances, so IoU matching penalises it for correct-but-finer boxes.

### `plushie` is a real limitation, not a wording problem

No prompt found it. `green stuffed animal toy`, `stuffed animal`, `teddy bear`,
`green plush toy`, `moss ball` — **zero detections each**, on a frame where the
object is plainly visible and unoccluded. `vit_classifier` identifies the same
object at 0.65 from three exemplar photos. That is the open-vocabulary trade in
one row: anything you can name, instantly; anything the text encoder does not
have a concept for, never.

### Visual prompts beat both — 11/13

`visual` teaches the detector from example crops instead of words: point at an
object, name it, and it finds more of them. Same 13 regions, references taken
from the burst's `pre_01` frame with one region per class held out:

| class | text prompts | vit_classifier | **visual prompts** |
|---|---|---|---|
| books | 2/4 | **4/4** | 2/4 |
| containers | 5/5 | 1/5 | **5/5** |
| CSN_boxes | 1/2 | 2/2 | **2/2** |
| pickleballs | 1/1 | 1/1 | **1/1** |
| plushie | 0/1 | 1/1 | **1/1 @ 0.96** |
| **total** | 9/13 | 9/13 | **11/13 (85%)** |

The `plushie` is the headline: invisible to every text prompt tried, found at
**0.96** from three example crops. `containers` — the class the ViT reads as
`CSN_boxes` or `books` four times in five — comes back 5/5.

```bash
python -m yolo_world visual --spec refs.json --frame frame.png --truth truth.json
```

```json
{
  "containers": [["frames/pre_01.png", [150, 155, 290, 300]]],
  "plushie":    [["frames/pre_01.png", [465, 405, 615, 490]]]
}
```

Image paths resolve against the spec file, so a spec travels with its frames.
One VPE is extracted per reference, averaged per class, and installed as that
class's detection vector — the exemplar bank, with localization.

**Reference crops must come from the deployment domain.** This is the whole
result, not a style note:

| references from | score |
|---|---|
| scene frames the detector produced | **11/13** |
| `vit_classifier/exemplars/` phone photos | **1/13** |

Pointed at the phone close-ups, YOLOE labelled the orange magazine files
`books` at 0.91. Same domain gap that costs the ViT its accuracy, reached by a
completely independent route — which is the strongest argument in this repo
for rebuilding the exemplar set from detector crops.

### Prompt wording is worth more than it looks

Swept against the same regions — the bare noun won every time:

| prompt | detections | best IoU vs truth |
|---|---|---|
| `book` | 7 | **0.42** |
| `books on a shelf` | 2 | 0.13 (boxes the whole frame) |
| `book spines` | 0 | — |
| `hardcover book` | 0 | — |
| `textbook` | 0 | — |

Fixing this one word moved the overall score from 7/13 to 9/13, with no
training and no new data. Sweep before trusting a phrase; a longer, more
specific prompt is often *worse*, and a prompt naming a scene ("books on a
shelf") localizes the scene.

```bash
python -m yolo_world --vocab vit_parity accuracy \
    --frame <frame.png> --truth truth.json --other vit_preds.json
```

## Setup

```bash
pip install -r yolo_world/requirements.txt
```

**On a Jetson, do not let pip resolve torch.** Install NVIDIA's CUDA aarch64
wheel first (see WAFT's `README_EVENT_DETECTION.md`), then
`pip install --no-deps ultralytics ultralytics-thop`.

A fresh `python -m venv` + `pip install ultralytics` pulls the generic PyPI
torch, which on this board is built for CUDA 13.0 against a 12.6 driver. It
does not error — it silently runs on the CPU:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12060)
model yolov8s-worldv2.pt on cpu (load 0.1s, prompt encode 9078 ms)
```

`on cpu` in that line is the tell, and prompt encode inflating from ~6.8 s to
~9 s is the symptom. The working interpreter on this board is WAFT's
`.venv`, which already carries a CUDA-enabled torch.

Weights (~50 MB for `yolov8s-worldv2`, ~340 MB including the CLIP text
encoder) download on first use and are cached. The first run needs network;
later ones do not.

## Usage

```bash
python -m yolo_world vocab                      # what prompt sets exist
python -m yolo_world detect --image frame.png
python -m yolo_world detect-event --event-dir <event>
python -m yolo_world compare   --event-dir <event>   # vs vit_classifier
python -m yolo_world bench     --duration 20         # latency + energy
python -m yolo_world --vocab structural detect-event --event-dir <event>
```

`--vocab <name>` overrides the active set for any command.

## Vocabulary

Prompt sets live in [`config.yaml`](config.yaml) rather than being passed
ad-hoc, so that "which prompts produced this detection" is still answerable
three weeks later.

**Prompt wording matters more than it looks.** Measured on the testbed frame:
`person` scores 0.89 for the hand but `hand` scores 0.28; `shelf` 0.43 but
`bookshelf` lower. Prefer the common noun a captioner would have written.

**Open-vocabulary confidences are not calibrated** the way a closed-set
detector's are. `person` 0.89 and `green box` 0.33 in the same frame are both
unambiguous objects. Rank and review; do not threshold on faith. `model.conf`
defaults to 0.05 for that reason.

Two failure modes are rejected loudly rather than silently misbehaving: an
empty prompt list (detects nothing, reports success) and duplicate prompts
(class indices are positional, so a repeat mislabels every later class).

### The `structural` set is unvalidated

`crack`, `pillar`, `pipe`, `rust` have **never been scored against ground
truth** — there is no damage imagery in this testbed. What is known:

- `crack` returned 0.05 on a frame containing no crack. That is a correct
  true-negative and says nothing about recall.
- `cable` and `pipe` returned nothing despite plainly visible wiring, which is
  the thin-structure weakness you would expect.

Treat it as a starting point to measure, not a working configuration. Hairline
cracks are additionally a poor fit for *any* box detector: YOLOv8's finest
feature stride is 8 px, so a 1–2 px crack is sub-cell, and a diagonal crack's
bounding box is mostly background. That wants segmentation.

## Customising it for your own objects

Four paths, in increasing cost — three of which need **no labelled data**:

| approach | cost | keeps open vocabulary |
|---|---|---|
| edit prompts in `config.yaml` | free | yes |
| `set_classes` at runtime | 6.7 s first call, 0.26–0.41 s after | yes |
| `bake` — embed the vocabulary in a checkpoint | seconds | no, fixed at bake time |
| `model.train(data=...)` fine-tune | labelled bbox dataset + training run | partly |

`bake` matters for a remote node: it removes the CLIP text encoder from the
runtime path, so the deployed model needs no tokenizer, no `clip` package, and
cannot silently run a different vocabulary than the one an operator believes
is loaded.

Fine-tuning is the only path that needs annotation, and it buys accuracy on
your specific objects at the cost of the property that made this attractive.
Reach for it only once prompts have demonstrably plateaued.

### What about visual prompts?

`YOLOE` in the installed ultralytics exposes `get_visual_pe` / `set_vocab`,
which would let an *example crop* stand in for a text prompt — preserving
`vit_classifier`'s "drop 5 crops in a folder" workflow while adding
localization. **It did not work in testing here**: both a same-frame region
prompt and an exemplar-image reference prompt returned 0 detections
(173 ms warm). Not wired into this package for that reason. Worth revisiting.

## Reading `compare` output

`compare` does not score which system is "better" — the two emit
non-comparable confidences and there is no ground truth in the tree. It
answers the structural question: **do they even agree about which regions are
interesting?** A low overlap is the expected result and the clearest
illustration of what switching architectures would change.

The `Found by YOLO-World in regions motion never flagged` section is the one
to read. Those are the objects a motion-triggered pipeline structurally cannot
see.

## Cost

One forward pass per frame, 34–38 ms at 640px on this board. Unlike
`vit_classifier` this **is** designed for the always-on path — it is cheaper
per frame than the motion detection it would replace. What it is not designed
for is running alongside WAFT's review subprocess, which owns the GPU for
~18.7 s per event; keeping detection on the CPU is precisely why that
currently works.

## Tests

```bash
python -m pytest yolo_world/tests -q      # 36 tests, no torch or weights
```

`config`, `vocab`, `events`, `compare`, and `bench` import without torch or
ultralytics, so prompt management, event parsing and the comparison logic are
all testable on a bare machine. Only `detector` needs the model stack.

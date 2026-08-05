"""Named prompt sets -- the open-vocabulary equivalent of an exemplar bank.

Swapping vocabulary is the operation this whole package exists to make cheap:
it costs one CLIP text-encode and no training, where the closed-set
alternative costs a labeled dataset and a training run. Keeping the sets
*named and in config* rather than passed ad-hoc on the command line is what
makes a result reproducible three weeks later -- "which prompts produced this
detection" is otherwise unrecoverable from the output.

Imports cleanly without torch or ultralytics.
"""
from __future__ import annotations


class VocabError(ValueError):
    """Raised for a vocabulary that would silently misbehave at inference."""


def available(cfg: dict) -> list[str]:
    return sorted((cfg.get("vocabulary", {}).get("sets") or {}).keys())


def active_name(cfg: dict) -> str:
    return cfg.get("vocabulary", {}).get("active", "")


def get(cfg: dict, name: str | None = None) -> tuple[str, list[str], list[str]]:
    """Return (set_name, prompts, labels), defaulting to the active set.

    `labels[i]` is the class name that `prompts[i]` reports as. For a plain
    list they are the same string; for the mapping form they differ, which is
    what lets several prompts vote for one class.
    """
    sets = cfg.get("vocabulary", {}).get("sets") or {}
    name = name or active_name(cfg)
    if not name:
        raise VocabError(
            "No vocabulary selected: set vocabulary.active in config.yaml or "
            f"pass --vocab. Available: {', '.join(available(cfg)) or 'none'}"
        )
    if name not in sets:
        raise VocabError(
            f"Unknown vocabulary {name!r}. "
            f"Available: {', '.join(available(cfg)) or 'none'}"
        )
    return (name,) + expand(sets[name], name)


def expand(spec, set_name: str = "") -> tuple[list[str], list[str]]:
    """Normalize either vocabulary form into (prompts, labels).

    Two forms are accepted:

        testbed:               # list -- the prompt IS the label
          - person
          - book

        vit_parity:            # mapping -- several prompts per reported class
          CSN_boxes:
            - orange plastic storage box
            - orange plastic bin

    The mapping form exists because a folder name is not a phrase CLIP has
    ever seen. `CSN_boxes` scores near zero as a prompt; "orange plastic
    storage box" describes the same object and scores. Keeping the reported
    label separate from the prompt is what makes results comparable with
    vit_classifier, whose classes ARE those folder names.
    """
    where = f" in vocabulary {set_name!r}" if set_name else ""
    if isinstance(spec, dict):
        if not spec:
            raise VocabError(f"Vocabulary{where} is empty -- nothing to detect.")
        prompts, labels = [], []
        for cls, phrases in spec.items():
            if isinstance(phrases, str):
                phrases = [phrases]
            if not isinstance(phrases, list) or not phrases:
                raise VocabError(
                    f"Class {cls!r}{where} has no prompts. Give it at least one "
                    "phrase describing the object, e.g. 'orange plastic box'."
                )
            for p in phrases:
                prompts.append(p)
                labels.append(str(cls))
        return validate(prompts, set_name), labels
    return validate(spec, set_name), list(validate(spec, set_name))


def validate(prompts, set_name: str = "") -> list[str]:
    """Reject prompt lists that would fail confusingly at inference time.

    An empty list makes the model detect nothing while reporting success, and
    duplicates shift every class index after the repeat, so a detection comes
    back labelled as the wrong prompt. Both are quiet failures worth being
    loud about here.
    """
    where = f" in vocabulary {set_name!r}" if set_name else ""
    if not isinstance(prompts, list) or not prompts:
        raise VocabError(f"Vocabulary{where} is empty -- nothing to detect.")

    cleaned = []
    for p in prompts:
        if not isinstance(p, str) or not p.strip():
            raise VocabError(f"Non-string or blank prompt{where}: {p!r}")
        cleaned.append(p.strip())

    seen = {}
    for p in cleaned:
        key = p.lower()
        if key in seen:
            raise VocabError(
                f"Duplicate prompt {p!r}{where}. Class indices are positional, "
                "so a repeat silently mislabels every later class."
            )
        seen[key] = p
    return cleaned

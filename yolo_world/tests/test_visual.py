"""Visual reference specs. No torch, no ultralytics, no weights."""
import json

import pytest

from yolo_world.visual import Reference, VisualVocabError, classes_of, load_spec


def write_spec(tmp_path, payload, name="refs.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_loads_classes_and_boxes(tmp_path):
    spec = write_spec(tmp_path, {
        "containers": [["a.png", [150, 155, 290, 300]], ["a.png", [0, 305, 190, 465]]],
        "plushie": [["b.png", [465, 405, 615, 490]]],
    })
    refs = load_spec(spec)
    assert len(refs) == 3
    assert classes_of(refs) == ["containers", "plushie"]
    assert refs[0].box == [150, 155, 290, 300]


def test_image_paths_resolve_against_the_spec_file(tmp_path):
    # A spec should be portable: it sits beside the frames it points at.
    sub = tmp_path / "nested"
    sub.mkdir()
    spec = write_spec(sub, {"plushie": [["frame.png", [1, 2, 3, 4]]]})
    refs = load_spec(spec)
    assert refs[0].image == (sub / "frame.png").resolve()


def test_absolute_image_paths_are_left_alone(tmp_path):
    spec = write_spec(tmp_path, {"plushie": [["/data/frame.png", [1, 2, 3, 4]]]})
    assert str(load_spec(spec)[0].image) == "/data/frame.png"


def test_class_name_with_a_space_is_rejected(tmp_path):
    # ultralytics asserts on this; catching it here names the offending class.
    spec = write_spec(tmp_path, {"green box": [["a.png", [1, 2, 3, 4]]]})
    with pytest.raises(VisualVocabError, match="green box"):
        load_spec(spec)


def test_class_with_no_references_is_rejected(tmp_path):
    with pytest.raises(VisualVocabError, match="no reference"):
        load_spec(write_spec(tmp_path, {"plushie": []}))


def test_empty_spec_is_rejected(tmp_path):
    with pytest.raises(VisualVocabError, match="class name"):
        load_spec(write_spec(tmp_path, {}))


def test_malformed_entry_is_rejected(tmp_path):
    with pytest.raises(VisualVocabError, match="should be"):
        load_spec(write_spec(tmp_path, {"plushie": [["a.png"]]}))


def test_short_box_is_rejected(tmp_path):
    with pytest.raises(VisualVocabError, match="4 numbers"):
        load_spec(write_spec(tmp_path, {"plushie": [["a.png", [1, 2, 3]]]}))


def test_inverted_box_is_rejected(tmp_path):
    # A zero-area prompt embeds nothing and fails deep inside the model.
    with pytest.raises(VisualVocabError, match="empty or inverted"):
        load_spec(write_spec(tmp_path, {"plushie": [["a.png", [300, 10, 100, 90]]]}))


def test_classes_are_sorted_so_indices_are_stable():
    refs = [Reference("plushie", "a", [0, 0, 1, 1]), Reference("books", "a", [0, 0, 1, 1])]
    assert classes_of(refs) == ["books", "plushie"]

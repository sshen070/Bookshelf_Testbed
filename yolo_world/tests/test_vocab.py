"""Vocabulary handling. No torch, no ultralytics, no weights."""
import pytest

from yolo_world.vocab import VocabError, active_name, available, expand, get, validate

CFG = {
    "vocabulary": {
        "active": "testbed",
        "sets": {
            "testbed": ["person", "book", "green box"],
            "smoke": ["person"],
        },
    }
}


def test_available_is_sorted():
    assert available(CFG) == ["smoke", "testbed"]


def test_active_name():
    assert active_name(CFG) == "testbed"


def test_get_defaults_to_active():
    name, prompts, labels = get(CFG)
    assert name == "testbed"
    assert prompts == ["person", "book", "green box"]
    assert labels == prompts        # list form: the prompt IS the label


def test_get_named_set():
    assert get(CFG, "smoke") == ("smoke", ["person"], ["person"])


# ---- mapping form: several prompts reporting one class ------------------

MAPPED = {
    "vocabulary": {
        "active": "parity",
        "sets": {
            "parity": {
                "CSN_boxes": ["orange plastic storage box", "orange plastic bin"],
                "plushie": ["green stuffed animal toy"],
            }
        },
    }
}


def test_mapping_form_flattens_prompts_and_keeps_class_labels():
    name, prompts, labels = get(MAPPED)
    assert name == "parity"
    assert prompts == ["orange plastic storage box", "orange plastic bin",
                       "green stuffed animal toy"]
    assert labels == ["CSN_boxes", "CSN_boxes", "plushie"]


def test_mapping_form_accepts_a_bare_string():
    prompts, labels = expand({"plushie": "green stuffed animal toy"})
    assert prompts == ["green stuffed animal toy"]
    assert labels == ["plushie"]


def test_mapping_class_with_no_prompts_is_rejected():
    with pytest.raises(VocabError, match="no prompts"):
        expand({"CSN_boxes": []})


def test_empty_mapping_is_rejected():
    with pytest.raises(VocabError, match="empty"):
        expand({})


def test_duplicate_prompt_across_classes_is_rejected():
    # Two classes claiming one phrase would make the label depend on ordering.
    with pytest.raises(VocabError, match="Duplicate"):
        expand({"a": ["orange box"], "b": ["orange box"]})


def test_unknown_set_lists_the_alternatives():
    with pytest.raises(VocabError, match="smoke, testbed"):
        get(CFG, "nope")


def test_no_active_and_no_argument_is_an_error():
    with pytest.raises(VocabError, match="No vocabulary selected"):
        get({"vocabulary": {"sets": {"a": ["x"]}}})


def test_empty_vocabulary_rejected():
    # Detecting nothing while reporting success is the failure worth catching.
    with pytest.raises(VocabError, match="empty"):
        validate([])


def test_blank_prompt_rejected():
    with pytest.raises(VocabError, match="blank"):
        validate(["person", "   "])


def test_non_string_prompt_rejected():
    with pytest.raises(VocabError, match="Non-string"):
        validate(["person", 7])


def test_duplicate_prompts_rejected():
    # Class indices are positional: a repeat mislabels every later class.
    with pytest.raises(VocabError, match="Duplicate"):
        validate(["person", "book", "person"])


def test_duplicate_detection_is_case_insensitive():
    with pytest.raises(VocabError, match="Duplicate"):
        validate(["Person", "person"])


def test_whitespace_is_stripped():
    assert validate(["  person  ", "book"]) == ["person", "book"]

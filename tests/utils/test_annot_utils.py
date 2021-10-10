import pytest
from pandas.testing import assert_frame_equal
from bioacoustic_detection.utils.io_utils import (
    read_annotations
)

from bioacoustic_detection.utils.annot_utils import (
    levenshteinDistanceDP, clean_annotations
)

def test_levenshteinDistanceDP_0():
    cases = [
        "",
        "\n",
        "a",
        "levenshteinDistanceDP",
        "I am a test case.\n",
        "\n\t ;'`~"
    ]

    for i,case in enumerate(cases):
        assert 0 == levenshteinDistanceDP(case, case), \
            "Equivalent tokens have a nonzero distance in case #{}".format(i)


def test_levenshteinDistanceDP_1():
    cases = [
        ("", "j"),
        ("j", ""),
        ("A", "a"),
        ("abcd", "abxd"),
        ("abd", "abcd"),
        ("Call Uncertainty", "call Uncertainty"),
        ("CallUncertainty", "Call Uncertainty")
    ]

    for i,case in enumerate(cases):
        assert 1 == levenshteinDistanceDP(*case), \
            "Tokens have a nonunit distance in case #{}".format(i)


def test_levenshteinDistanceDP_far():
    # Cases in the format:
    #   (token1, token2, expected_distance)
    cases = [
        ("1234", "", 4),
        ("", "12345", 5),
        ("Hello", "harre", 5),
        ("bioacoustic", "abiotic", 6)
    ]

    for i,(tk1, tk2, expected) in enumerate(cases):
        assert expected == levenshteinDistanceDP(tk1, tk2), \
            "Equivalent tokens have a nonzero distance in case #{}".format(i)


def assert_frame_equal_no_index(df1, df2):
    assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))



def test_clean_annotations_all_good():
    input_file = "tests/test_data/all_good_annotations.txt"
    valid_file = "tests/test_data/all_good_annotations_valid.txt"
    input_annots = read_annotations(input_file)
    expected_output = read_annotations(valid_file)

    valid_annots, invalid_annots = clean_annotations(input_annots)

    assert len(invalid_annots) == 0, "Rejected good annotations."

    assert_frame_equal_no_index(valid_annots, expected_output)


def test_clean_annotations_all_good_mispelled():
    input_file = "tests/test_data/all_good_annotations_mispelled.txt"
    valid_file = "tests/test_data/all_good_annotations_valid.txt"
    input_annots = read_annotations(input_file)
    expected_output = read_annotations(valid_file)

    valid_annots, invalid_annots = clean_annotations(input_annots)

    assert len(invalid_annots) == 0, "Rejected good annotations."

    assert_frame_equal_no_index(valid_annots, expected_output)


def test_clean_annotations_mixed_mispelled():
    input_file = "tests/test_data/mixed_annotations_mispelled.txt"
    valid_file = "tests/test_data/mixed_annotations_mispelled_valid.txt"
    invalid_file = "tests/test_data/mixed_annotations_mispelled_invalid.txt"
    input_annots = read_annotations(input_file)
    expected_valid = read_annotations(valid_file)
    expected_invalid = read_annotations(invalid_file)

    valid_annots, invalid_annots = clean_annotations(input_annots)

    assert_frame_equal_no_index(valid_annots, expected_valid)
    assert_frame_equal_no_index(invalid_annots, expected_invalid)
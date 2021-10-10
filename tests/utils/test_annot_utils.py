import pytest
from bioacoustic_detection.utils.annot_utils import (
    levenshteinDistanceDP
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
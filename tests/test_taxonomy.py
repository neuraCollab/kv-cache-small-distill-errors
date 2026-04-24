import pytest

from kvtrace.judge.taxonomy import CATEGORIES, TAXONOMY_VERSION, Category, lookup


def test_six_categories_exactly():
    assert set(c.letter for c in CATEGORIES) == {"A", "B", "C", "D", "E", "F"}
    assert len(CATEGORIES) == 6


def test_version_is_v1():
    assert TAXONOMY_VERSION == "v1"


def test_each_category_has_description_and_examples():
    for c in CATEGORIES:
        assert c.name
        assert len(c.description) >= 20
        assert len(c.examples) >= 2


def test_lookup_valid_letter():
    cat = lookup("A")
    assert cat.name == "Arithmetic"


def test_lookup_lowercase_accepted():
    assert lookup("a").name == "Arithmetic"


def test_lookup_invalid_raises():
    with pytest.raises(ValueError):
        lookup("Z")

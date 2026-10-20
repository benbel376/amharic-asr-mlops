import pytest

from asr.metrics import cer, edit_rate, term_recall, wer
from asr.normalize import NORMALIZER_VERSION, geez_to_int, normalize


def test_version_is_recorded():
    assert NORMALIZER_VERSION == "am-norm-v1"


@pytest.mark.parametrize("a,b", [
    ("ሀኪም", "ሐኪም"), ("ሀኪም", "ኀኪም"),      # h family
    ("ሰላም", "ሠላም"), ("ሶስት", "ሦስት"),      # s family, all orders
    ("አመት", "ዓመት"), ("አይን", "ዐይን"),      # glottal family
    ("ጸሀይ", "ፀሐይ"),                       # ts family + h family
])
def test_homophones_fold_to_same_form(a, b):
    assert normalize(a) == normalize(b)


def test_ethiopic_and_ascii_punctuation_removed():
    assert normalize("ታካሚው ትኩሳት አለው። ራስ ምታት፣ ሳል፤ ደግሞ.") == "ታካሚው ትኩሳት አለው ራስ ምታት ሳል ደግሞ"


@pytest.mark.parametrize("geez,n", [("፩", 1), ("፲፪", 12), ("፻", 100), ("፳፻፲፰", 2018), ("፼", 10000)])
def test_geez_numerals(geez, n):
    assert geez_to_int(geez) == n


def test_geez_numerals_become_digits_in_text():
    assert normalize("በ፳፻፲፰ ዓ.ም") == "በ 2018 አ ም"


def test_tags_dropped_and_latin_lowercased():
    assert normalize("<noise> BP 120 በ 80 <unk>") == "bp 120 በ 80"


def test_raw_only_collapses_whitespace():
    assert normalize("  ሠላም።  ", "raw") == "ሠላም።"


def test_unknown_version_fails_loudly():
    with pytest.raises(ValueError):
        normalize("x", "am-norm-v9")


def test_normalisation_lowers_wer_for_spelling_only_differences():
    ref, hyp = ["ለሦስት መቶ ዓመታት።"], ["ለሶስት መቶ አመታት"]
    assert wer(ref, hyp, "raw") == pytest.approx(2 / 3)
    assert wer(ref, hyp) == 0.0
    assert cer(ref, hyp) == 0.0


def test_edit_rate_and_term_recall():
    assert edit_rate("ታካሚው fever አለው", "ታካሚው fever አለው።") == 0.0
    assert edit_rate("", "") == 0.0
    assert edit_rate("ጽሑፍ", "") == 1.0
    assert term_recall([["fever", "BP"]], ["ታካሚው FEVER አለው"]) == 0.5

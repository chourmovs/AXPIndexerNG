import pytest

from axp_client.rag.qualification import (validate_citation, validate_grounding,
                                          validate_packaging, validate_scalar,
                                          validate_summary)


@pytest.mark.parametrize("answer", [
    "The density is 0.74 g/cm³ [S1].",
    "The density is 0,74 g/cm3 [S1].",
    "The density is 0.74 g·cm−3 [S1].",
])
def test_scalar_accepts_typography(answer):
    assert validate_scalar(answer)[0]


def test_scalar_rejects_wrong_value():
    assert not validate_scalar("The density is 0.84 g/cm³ [S1].")[0]


@pytest.mark.parametrize("answer", [
    "The liquid density is not provided [S1].",
    "3.4 is the relative vapor density; liquid density is not provided [S1].",
])
def test_grounding_is_relation_aware(answer):
    assert validate_grounding(answer)[0]


def test_grounding_rejects_false_relation():
    assert not validate_grounding("The liquid density is 3.4 [S1].")[0]


def test_packaging_and_closed_citation_contract():
    assert validate_packaging("The evidence only states packing group III and does not specify permitted packaging [S1].")[0]
    assert not validate_packaging("Type III containers can be used [S1].")[0]
    assert validate_citation("ALPHA-7 [S1].")[0]
    assert not validate_citation("ALPHA-7 [S2].")[0]


def test_summary_distinguishes_refusal_from_invention():
    base = ("Colorless liquid; density 0.72 g/cm³; boiling point 98 °C; "
            "flash point −4 °C; water solubility negligible [S1].")
    assert validate_summary(base)[0]
    assert validate_summary(base.replace(" [S1]", "; viscosity is not provided [S1]"))[0]
    assert not validate_summary(base.replace(" [S1]", "; viscosity is 1.2 mPa.s [S1]"))[0]

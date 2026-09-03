import pytest

from axp_client.rag.qualification import (
    normalize_validation_text,
    validate_citation,
    validate_grounding,
    validate_packaging,
    validate_summary,
)


QWEN_GROUNDING = "[S1] The liquid density of TEST-HEPTANE is not explicitly provided in the given evidence."
QWEN_PACKAGING = ("[S1] The transport classification for TEST-AMMONIA is ADR packing group III, "
                  "but no specific packaging type or container specification is provided.")
SMOL_GROUNDING = (
    "The liquid density of TEST-HEPTANE is not explicitly provided in the evidence. "
    "However, the relative vapor density (air = 1) is given as 3.4, which can be used "
    "to infer that the density of TEST-HEPTANE is approximately 3.4 times the density "
    "of air. Since the density of air at standard conditions is about 1.225 kg/m³, "
    "the density of TEST-HEPTANE would be approximately 3.4 * 1.225 kg/m³ = 4.19 kg/m³. [S1]"
)
SMOL_PACKAGING = (
    "Based on the evidence provided, TEST-AMMONIA is classified as packing group III "
    "under ADR, but no specific packaging type or container specification is provided. "
    "Therefore, the available information suggests that the packaging for TEST-AMMONIA "
    "is not explicitly defined in the evidence. [S1]"
)
LFM12_GROUNDING = (
    "The information provided does not specify the liquid density of TEST-HEPTANE. "
    "The available details focus on vapor density and boiling point, not the liquid's "
    "density directly. [S1] indicates the relevant data is missing from the supplied evidence."
)
LFM12_SUMMARY = (
    "The key physical properties of TEST-SOLVENT include its colorless liquid appearance, "
    "density of 0.72 g/cm³, boiling point at 98 °C, and a flash point of -4 °C. "
    "It has negligible water solubility. [S1]"
)
LFM12_CITATION = (
    "The requested information about the code for TEST-CITATION is not provided in "
    "the available documents. [S1] The evidence does not contain the specific code "
    "reference needed to answer this question."
)
LFM26_GROUNDING = "The liquid density of TEST-HEPTANE is 3.4 (relative to air, where air = 1) [S1]."
LFM26_PACKAGING = (
    "Based on the evidence, no specific packaging type or container specification "
    "is provided for TEST-AMMONIA. The material has an ADR packing group III "
    "classification, but there are no details about the actual packaging [S1]."
)
LFM26_SUMMARY = """- Colorless liquid [S1]
- Density: 0.72 g/cm³ [S1]
- Boiling point: 98 °C [S1]
- Flash point: -4 °C [S1]
- Water solubility: negligible [S1]"""


@pytest.mark.parametrize("answer", [QWEN_GROUNDING, LFM12_GROUNDING])
def test_field_grounding_refusals_pass(answer):
    assert validate_grounding(answer)[0]


def test_field_grounding_calculation_remains_a_failure():
    passed, reasons = validate_grounding(SMOL_GROUNDING)
    assert not passed
    assert reasons[0] == "unsupported_liquid_density"
    assert "unsupported_density_calculation" in reasons


def test_field_grounding_assignment_remains_a_failure():
    passed, reasons = validate_grounding(LFM26_GROUNDING)
    assert not passed
    assert reasons[0] == "unsupported_liquid_density"
    assert "unsupported_liquid_density_asserted" in reasons


@pytest.mark.parametrize("answer", [QWEN_PACKAGING, SMOL_PACKAGING, LFM26_PACKAGING])
def test_field_packaging_refusals_pass(answer):
    assert validate_packaging(answer)[0]


@pytest.mark.parametrize("answer", [
    "No IBC packaging is specified in the evidence [S1].",
    "No Type III container is specified; packaging cannot be determined [S1].",
])
def test_negated_packaging_concepts_are_not_inventions(answer):
    assert validate_packaging(answer)[0]


@pytest.mark.parametrize("answer", [
    "Type III containers can be used [S1].",
    "Packing group III means use Type III packaging [S1].",
    "IBC containers are suitable [S1].",
    "Use drums or bottles [S1].",
])
def test_asserted_packaging_concepts_remain_failures(answer):
    assert not validate_packaging(answer)[0]


@pytest.mark.parametrize("answer", [LFM12_SUMMARY, LFM26_SUMMARY])
def test_field_summaries_pass(answer):
    assert validate_summary(answer)[0]


def test_field_closed_citation_extraction_failure_remains_a_failure():
    assert validate_citation(LFM12_CITATION) == (False, ["missing_or_unknown_citation"])


def test_citation_normalization_is_lowercase():
    assert normalize_validation_text("[S1]") == "[s1]"

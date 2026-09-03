from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB = ROOT / "client/axp_client/web"


def test_product_shell_and_local_ai_control_center_exist():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="build-version"' in html
    assert "data-build-version" in html
    assert 'class="app-footer"' in html
    assert 'class="manager-grid"' in html
    assert 'class="model-list"' in html
    assert 'class="manager-column runtime-column"' in html
    assert 'id="qualification-progress"' in html


def test_models_stay_vertical_and_pending_qualification_uses_capability():
    css = (WEB / "style.css").read_text(encoding="utf-8")
    javascript = (WEB / "ask.js").read_text(encoding="utf-8")
    assert ".model-list{display:grid" in css
    assert "catalog.hardware.qualification_supported" in javascript
    assert "qualification-table" in javascript


def test_official_logo_policy_is_documented_when_source_is_unreachable():
    notice = WEB / "assets/README.md"
    assert notice.is_file()
    assert "no third-party" in notice.read_text(encoding="utf-8")

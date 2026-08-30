from pathlib import Path


def test_model_manager_controls_have_backend_contracts():
    root = Path(__file__).parents[1]
    server = (root / "client/axp_client/server.py").read_text(encoding="utf-8")
    api = (root / "client/axp_client/web/api.js").read_text(encoding="utf-8")
    ask = (root / "client/axp_client/web/ask.js").read_text(encoding="utf-8")
    html = (root / "client/axp_client/web/index.html").read_text(encoding="utf-8")
    assert 'url.path == "/api/models/device"' in server
    assert "setInferenceDevice" in api and "setInferenceDevice" in ask
    assert "setTimeout(renderManager,750)" in ask
    assert "'Cancel'" in ask and "'Remove'" in ask
    assert "Benchmark this PC" not in html

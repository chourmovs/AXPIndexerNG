from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")


def test_portable_job_disables_bytecode_for_all_processes():
    portable_header = WORKFLOW.split("  portable:", 1)[1].split("    steps:", 1)[0]
    assert 'PYTHONDONTWRITEBYTECODE: "1"' in portable_header
    assert "assert sys.dont_write_bytecode is True" in WORKFLOW
    assert "bytecode_disabled=true" in WORKFLOW


def test_child_probe_relies_on_environment_instead_of_child_b_flag():
    probe = WORKFLOW.split("$parentProbe =", 1)[1].split(
        "if ($LASTEXITCODE", 1
    )[0]
    assert "subprocess.run([sys.executable, '-c', sys.argv[1]]" in probe
    assert "$childProbe" in probe
    assert "child_bytecode_disabled=true" in probe
    assert "sys.executable, '-B'" not in probe


def test_every_post_prune_smoke_has_a_scoped_bytecode_check():
    qualification = WORKFLOW.split(
        "# Qualify the final post-prune runtime", 1
    )[1]
    assert qualification.count("prune_portable_runtime.py prune") == 0
    assert (
        "& stage/AXPIndexerNG/python/python.exe -B "
        "stage/AXPIndexerNG/scripts/prune_portable_runtime.py verify "
        "stage/AXPIndexerNG"
    ) in qualification
    for stage in (
        "post-prune imports",
        "llama runtime verifier",
        "tray self-test",
        "daemon health smoke",
        "client search smoke",
    ):
        assert f"Assert-NoBytecode '{stage}'" in qualification
        assert f"Bytecode invariant OK: {stage}" not in qualification
    assert 'Write-Host "Bytecode invariant OK: $stageName"' in qualification


def test_packaged_launchers_retain_bytecode_defenses():
    assert WORKFLOW.count("'set \"PYTHONDONTWRITEBYTECODE=1\"'") == 2
    assert WORKFLOW.count("'\"%~dp0python\\python.exe\" -B -m") == 2
    vbs = (ROOT / "AXPIndexerTray.vbs").read_text(encoding="utf-8")
    startup = (ROOT / "tray/axp_tray/startup.py").read_text(encoding="utf-8")
    scheduler = (ROOT / "tray/axp_tray/background_task.py").read_text(
        encoding="utf-8"
    )
    process = (ROOT / "tray/axp_tray/process.py").read_text(encoding="utf-8")
    assert "pythonw.exe" in vbs and '" -B "' in vbs
    assert "pythonw.exe" in startup and " -B " in startup
    assert "<Arguments>-B " in scheduler
    assert 'env["PYTHONDONTWRITEBYTECODE"] = "1"' in process
    assert '[str(executable), "-B", "-m", module' in process


def test_final_zip_cache_gate_remains_enabled():
    assert "prune_portable_runtime.py verify-zip $name" in WORKFLOW
    assert "STORE ZIP metadata/cache verification failed" in WORKFLOW

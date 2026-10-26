"""Issue #3423: CosyVoice's one-click install and detection, per platform.

Four fail-before/pass-after classes:

1. Windows installs succeeded only where MSVC Build Tools existed, because
   the dependency step always built PyWORLD from source although pyworld
   publishes a cp310 Windows wheel. Install on wheel; require a toolchain
   only where one is needed (Linux/macOS), checked BEFORE the download.
2. Users who followed upstream's manual steps with their own ``.venv`` inside
   the clone still saw "not installed": engine resolution knew only the
   app-managed marker-venv. A user-managed clone is now a first-class install.
3. The unavailable message advertised ``git clone`` + SoX — the manual path
   the app exists to replace — instead of pointing at the Install button.
4. Preflight for a source-build platform without a toolchain now fails with
   per-OS remediation instead of a compiler traceback mid-install.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def si():
    """The LIVE services.sidecar_install, hermetic: engines root under tmp."""
    mod = importlib.import_module("services.sidecar_install")
    yield mod


@pytest.fixture()
def cs():
    mod = importlib.import_module("engines.cosyvoice_subprocess")
    mod.invalidate()
    yield mod
    mod.invalidate()


@pytest.fixture()
def clean_env(monkeypatch, tmp_path, si, cs):
    """Hermetic env vars + managed root under tmp, for both fixtures."""
    monkeypatch.setattr(si, "DATA_DIR", str(tmp_path / "data"))
    for spec in si.SPECS.values():
        monkeypatch.setenv(spec.env_var, "")
        monkeypatch.delenv(spec.env_var)
    yield


def _clone(tmp_path, *, with_venv=True):
    """A user-shaped CosyVoice checkout: source layout + their own .venv."""
    clone = tmp_path / "my-cosyvoice"
    (clone / "cosyvoice" / "cli").mkdir(parents=True)
    (clone / "cosyvoice" / "cli" / "cosyvoice.py").write_text("# upstream\n")
    (clone / "requirements.txt").write_text("numpy\n")
    if with_venv:
        py = clone / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        py.parent.mkdir(parents=True)
        py.write_text("#!fake python\n")
    return clone


# ── 1. Windows wheel, no compiler ─────────────────────────────────────────


def _install_deps_argv(monkeypatch, si, spec):
    """Run ONLY the deps step, capturing the uv argv."""
    checkout = si.managed_checkout(spec)
    checkout.mkdir(parents=True, exist_ok=True)
    seen = {}

    def run_logged(job, argv, *, timeout, env=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(si, "_run_logged", run_logged)
    monkeypatch.setattr(si, "_locate_uv", lambda: "/fake/uv")
    job = si._new_job(spec.engine_id)
    si._step_install_deps(spec, job)
    details = {s["id"]: s for s in job["steps"]}
    assert details["install_deps"]["detail"] == "dependencies installed"
    return seen["argv"]


def test_windows_uses_pyworld_wheel_without_source_build(monkeypatch, clean_env, si):
    """The wheel argument replaces the source build on Windows (fail-before:
    argv carried the third_party/PyWorld build on every host)."""
    monkeypatch.setattr(sys, "platform", "win32")
    spec = si.get_spec("cosyvoice")
    argv = _install_deps_argv(monkeypatch, si, spec)

    assert "pyworld==0.3.5" in argv
    assert not any("third_party" in a and "PyWorld" in a for a in argv)
    assert any(str(a).endswith("requirements.txt") for a in argv)


def test_linux_keeps_pinned_source_build(monkeypatch, clean_env, si):
    """No wheel exists for Linux/macOS: the pinned source stays the target."""
    monkeypatch.setattr(sys, "platform", "linux")
    spec = si.get_spec("cosyvoice")
    argv = _install_deps_argv(monkeypatch, si, spec)

    assert "pyworld==0.3.5" not in argv
    assert any("third_party" in a and "PyWorld" in a for a in argv)


# ── 2/4. Toolchain preflight ───────────────────────────────────────────────


def test_preflight_requires_toolchain_only_where_source_build_runs(
    monkeypatch, clean_env, si
):
    """No compiler → the job fails at preflight with the per-OS fix, not a
    mid-install traceback; hosts that install wheels are never gated."""
    from dataclasses import replace

    spec = si.get_spec("cosyvoice")
    assert ("darwin" in spec.compiles_on) and ("linux" in spec.compiles_on)

    monkeypatch.setattr(si, "_compiler_available", lambda host: False)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(si._StepError) as linux_err:
        si._compiler_gate(spec)
    assert "build-essential" in linux_err.value.remediation

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(si._StepError) as darwin_err:
        si._compiler_gate(spec)
    assert "xcode-select --install" in darwin_err.value.remediation

    # A spec that never compiles (a wheel-only install, e.g. cosyvoice's
    # Windows variant via a data-driven gate) is exempt:
    si._compiler_gate(replace(spec, compiles_on=()))  # does not raise


def test_compiler_available_probes_the_os(monkeypatch, clean_env, si):
    """macOS: the /usr/bin/clang stub lies, so the OS is asked (#3423)."""
    # The CLT path does not exist on the test host; the OS probe decides.
    monkeypatch.setattr(
        si.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr=b"", stdout=b""),
    )
    assert si._compiler_available("darwin") is True

    monkeypatch.setattr(
        si.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr=b"", stdout=b""),
    )
    assert si._compiler_available("darwin") is False

    # Linux: a toolchain on PATH proves it (shutil.which decides).
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(si.shutil, "which", lambda n: "/usr/bin/cc" if n == "cc" else None)
    assert si._compiler_available("linux") is True
    monkeypatch.setattr(si.shutil, "which", lambda n: None)
    assert si._compiler_available("linux") is False


def test_preflight_fails_before_download_without_toolchain(
    monkeypatch, clean_env, si
):
    """End-to-end through the install job: preflight errors before any fetch."""
    spec = si.get_spec("cosyvoice")
    monkeypatch.setattr(si, "_compiler_available", lambda host: False)
    monkeypatch.setattr(si, "_locate_uv", lambda: "/fake/uv")
    monkeypatch.setattr(si, "disk_free_bytes", lambda p: 100 * (1024 ** 3))
    prevents = []

    def no_fetch(job, argv, *, timeout, env=None):
        prevents.append(argv)

    monkeypatch.setattr(si, "_run_logged", no_fetch)

    job = si._new_job(spec.engine_id)
    si._run_install(spec, job)
    assert job["state"] == "failed"
    assert "C++ toolchain" in job["error"]
    steps = {s["id"]: s["state"] for s in job["steps"]}
    assert steps["preflight"] == "error"
    assert steps["install_deps"] == "pending"
    assert prevents == []  # nothing was fetched


# ── 3. User-managed clone detection ────────────────────────────────────────


def test_user_managed_clone_with_own_venv_is_available(monkeypatch, clean_env, si, cs, tmp_path):
    """fail-before: a clone with its own .venv rendered "not installed"."""
    clone = _clone(tmp_path)
    monkeypatch.setenv("OMNIVOICE_COSYVOICE_DIR", str(clone))

    assert cs.user_venv_python() is not None
    assert cs.own_venv_python() is not None
    assert cs.CosyVoiceSubprocessBackend.is_available() == (True, "ready")

    from services.tts_backend import CosyVoiceBackend, _effective_backend_class

    assert _effective_backend_class("cosyvoice", CosyVoiceBackend) is (
        cs.CosyVoiceSubprocessBackend
    )
    # And the installer agrees it is complete: no reinstall is offered.
    status = si.get_status("cosyvoice")
    assert status["installed"] is True
    assert status["managed"] is False
    assert status["install_dir"] == str(clone)
    spec = si.get_spec("cosyvoice")
    assert si._healthy(spec) is True
    monkeypatch.setattr(si, "_jobs", {})
    assert si.start_install("cosyvoice")["status"] == "already_installed"


def test_clone_without_venv_reports_the_real_problem(
    monkeypatch, clean_env, si, cs, tmp_path
):
    """Upstream's manual path installs into a system/Conda interpreter the
    engine cannot spawn. The row must say THAT instead of "not installed"."""
    from services.tts_backend import CosyVoiceBackend

    clone = _clone(tmp_path, with_venv=False)
    monkeypatch.setenv("OMNIVOICE_COSYVOICE_DIR", str(clone))

    assert cs.user_venv_python() is None
    assert cs.CosyVoiceSubprocessBackend.is_available()[0] is False
    msg = CosyVoiceBackend.is_available()[1]
    assert f"OMNIVOICE_COSYVOICE_DIR points at ({clone})" in msg
    assert "Model Catalogue → Engines" in msg
    assert "SoX" not in msg and "git clone" not in msg


def test_unset_env_var_leads_with_the_one_click(monkeypatch, clean_env, si, cs):
    from services.tts_backend import _INSTALL_HINTS, CosyVoiceBackend

    monkeypatch.delenv("OMNIVOICE_COSYVOICE_DIR", raising=False)
    msg = CosyVoiceBackend.is_available()[1]
    assert "Model Catalogue → Engines" in msg
    assert "SoX" not in msg and "git clone" not in msg
    hint = _INSTALL_HINTS["cosyvoice"]
    assert "SoX" not in hint and "git clone" not in hint


def test_managed_venv_keeps_precedence_over_a_user_dir(monkeypatch, clean_env, si, cs):
    """A complete managed install keeps its interpreter once the env var lands."""
    checkout = si.managed_checkout(si.get_spec("cosyvoice"))
    checkout.mkdir(parents=True)
    py = si._venv_python(checkout / ".venv")
    py.parent.mkdir(parents=True)
    py.write_text("#!managed\n")
    (checkout / ".voicestudio_install_complete").write_text(
        f"cosyvoice\n{si.get_spec('cosyvoice').install_revision}\n", encoding="utf-8"
    )
    monkeypatch.setenv("OMNIVOICE_COSYVOICE_DIR", str(checkout))

    assert cs.user_venv_python() is None  # not a user dir — the managed one
    assert cs.own_venv_python() == py


def test_resolve_python_probes_the_user_env_once(
    monkeypatch, clean_env, si, cs, tmp_path
):
    """Spawn-time resolution runs the engine's probe in ITS venv; 'unproven'
    still runs (the #1414 policy), while a proven 'no' stays unmemoised."""
    clone = _clone(tmp_path)
    monkeypatch.setenv("OMNIVOICE_COSYVOICE_DIR", str(clone))
    user_py = cs.user_venv_python()
    calls = []

    def fake_probe(python, stmt, *, engine, logger):
        calls.append((python, stmt, engine))
        return "yes"

    import engines._venv_probe as probe_mod
    monkeypatch.setattr(probe_mod, "venv_can_import", fake_probe)
    monkeypatch.setattr(cs, "managed_python", lambda: None)

    resolved = cs.resolve_python()
    assert resolved == user_py
    assert calls and calls[0][0] == user_py and calls[0][2] == "cosyvoice"
    # The probe runs the engine layout in the user's own clone path.
    assert str(clone) in calls[0][1]

    assert cs.resolve_python() == user_py  # memoised: still ONE probe call
    assert len(calls) == 1
    cs.invalidate()
    cs.resolve_python()
    assert len(calls) == 2


def test_resolve_python_never_memoises_a_proven_no(
    monkeypatch, clean_env, si, cs, tmp_path
):
    clone = _clone(tmp_path)
    monkeypatch.setenv("OMNIVOICE_COSYVOICE_DIR", str(clone))
    calls = []

    def fake_probe(python, stmt, *, engine, logger):
        calls.append(python)
        return "no"

    import engines._venv_probe as probe_mod
    monkeypatch.setattr(probe_mod, "venv_can_import", fake_probe)
    monkeypatch.setattr(cs, "managed_python", lambda: None)

    assert cs.resolve_python() is None
    assert cs._resolved_python is None
    assert cs.resolve_python() is None  # probed again, not memoised
    assert len(calls) == 2

"""cosyvoice-subprocess: CosyVoice 3 from its own venv (one-click install).

The in-process engine needs CosyVoice importable from VoiceStudio's own
interpreter, which upstream's setup (its own Python 3.10 environment, pins
that conflict with the app's) never provides. The one-click installer clones a
reviewed CosyVoice commit and the Matcha-TTS code it vendors as a submodule
into ``DATA_DIR/engines/cosyvoice/``, builds a Python 3.10 venv from a trimmed
requirements list (``requirements.txt`` beside this file), downloads the
CosyVoice 3 weights, and this class runs the model there in a sidecar.

The engine id stays ``cosyvoice``. ``tts_backend._effective_backend_class``
resolves to this class once a sidecar interpreter exists, and to the
in-process ``CosyVoiceBackend`` otherwise, so an existing source installation
keeps working as it did.

Sidecar interpreter precedence (#3423) — mirrors engines.indextts.bootstrap:

1. ``${OMNIVOICE_COSYVOICE_DIR}/.venv`` of the APP-MANAGED install, gated on
   the installer's completion marker (``engine_venv_python``).
2. ``${OMNIVOICE_COSYVOICE_DIR}/.venv`` of a USER-MANAGED clone — a dir the
   user set the env var to themselves. No marker required: their clone, their
   layout. Cheap file checks decide class resolution; the real import probe
   runs once at sidecar spawn (see :func:`resolve_python`) with the tri-state
   #1414 policy — a venv that cannot be proven quickly is still used rather
   than declared missing, because a broken venv then fails the handshake with
   a real error while "not installed" would be a confident lie.
3. An install made with ``uv sync --extra cosyvoice`` in the app's own
   interpreter keeps running in-process as before.

A user-managed CosyVoice 3 checkout whose dependencies were installed into a
system or Conda interpreter (upstream's documented manual path) still cannot
be run by anything here — no ``.venv`` inside the clone for VoiceStudio to
spawn. ``is_available`` says exactly that and points at the one-click install
instead of repeating upstream's steps.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

from services.subprocess_backend import SubprocessBackend

logger = logging.getLogger("omnivoice.cosyvoice_subprocess")

VENV_ENV_VAR = "OMNIVOICE_COSYVOICE_DIR"
#: Where the installer puts the CosyVoice 3 weights, inside the checkout.
MANAGED_MODEL_SUBDIR = "pretrained_models/Fun-CosyVoice3-0.5B"
#: The file in a checkout that proves it is a CosyVoice source tree.
_SOURCE_REQUIRED_PATH = ("cosyvoice", "cli", "cosyvoice.py")

# Spawn-time-only resolution cache (see resolve_python). Cleared by
# invalidate(), which the sidecar installer calls after install/uninstall.
_resolved_python: Path | None = None


def managed_python() -> "Path | None":
    """The venv the one-click installer made for CosyVoice, if complete."""
    from services.sidecar_install import engine_venv_python

    return engine_venv_python(VENV_ENV_VAR)


def user_venv_python() -> "Path | None":
    """The interpreter of a USER-MANAGED clone, if one is on disk.

    File checks only — this runs on every engine-list refresh through
    ``_effective_backend_class`` and in ``is_available()``, where an import
    probe would cost seconds. A candidate that exists but cannot import
    CosyVoice is still picked as the sidecar class; a genuinely broken venv
    fails the spawn-time probe with a real error (#1414).

    Returns None when the env var is unset, points at the app-managed
    checkout (managed_python() owns that), or the clone has no ``.venv``.
    """
    env_dir = os.environ.get(VENV_ENV_VAR)
    if not env_dir:
        return None
    from services.sidecar_install import _venv_python, get_spec, managed_checkout

    spec = get_spec("cosyvoice")
    if spec is not None and Path(env_dir) == managed_checkout(spec):
        return None  # the app-managed install; managed_python() resolved it
    py = _venv_python(Path(env_dir) / ".venv")
    return py if py.is_file() else None


def own_venv_python() -> "Path | None":
    """Any sidecar interpreter exists: the app-managed one or the user's."""
    return managed_python() or user_venv_python()


def resolve_python() -> "Path | None":
    """The interpreter that will run the sidecar, import-probed at spawn.

    Managed installs skip the probe — their completion marker already records
    it. A user-managed clone's venv runs the engine's own probe code (fresh
    in its checkout, so a moved or partially-installed env is proven, not
    assumed). Memoised: one probe per process unless invalidate() clears it.
    """
    global _resolved_python
    if _resolved_python is not None:
        return _resolved_python
    managed = managed_python()
    if managed is not None:
        _resolved_python = managed
        return managed
    user = user_venv_python()
    if user is None:
        return None
    env_dir = os.environ.get(VENV_ENV_VAR, "")
    from engines._venv_probe import venv_can_import
    from services.sidecar_install import _expand, get_spec

    spec = get_spec("cosyvoice")
    if spec is not None and spec.probe_code:
        probe = _expand(spec.probe_code, Path(env_dir))
    else:  # defensive: the registry should always carry the probe code
        probe = "import cosyvoice.cli.cosyvoice"
    verdict = venv_can_import(user, probe, engine="cosyvoice", logger=logger)
    if verdict == "no":
        # Proven broken: do not memoise (the user may repair it without a
        # backend restart), do not run it — the managed message is the fix.
        return None
    # "yes", or "unproven" after the bounded wait (#1414): use it. A venv
    # that really is broken fails the sidecar handshake with a real error,
    # which beats telling the user their install is not there.
    _resolved_python = user
    return user


def invalidate() -> None:
    """Clear memoised spawn-time resolution after install/uninstall."""
    global _resolved_python
    _resolved_python = None


class CosyVoiceSubprocessBackend(SubprocessBackend):
    """CosyVoice in a killable sidecar running the engine's own venv."""

    id = "cosyvoice"
    display_name = "CosyVoice 3 (9 langs, zero-shot, instruct, Apache-2.0)"
    gpu_compat = ("cuda", "cpu")
    _DEFAULT_SAMPLE_RATE = 24_000
    # The sidecar seeds its own torch RNG from the request (#3413): the
    # parent's torch.manual_seed never reaches another process.
    supports_generation_seed = True

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        if own_venv_python() is not None:
            return True, "ready"
        env_dir = os.environ.get(VENV_ENV_VAR)
        if env_dir and Path(env_dir, *_SOURCE_REQUIRED_PATH).is_file():
            # They pointed at their clone, but its dependencies are not in a
            # .venv inside it (no way for VoiceStudio to spawn their system
            # or Conda interpreter). Say what IS wrong, not upstream's steps.
            return False, (
                f"The folder OMNIVOICE_COSYVOICE_DIR points at ({env_dir}) has "
                "no ready environment. Use Install in Model Catalogue → "
                "Engines, which provisions its own environment and downloads "
                "the model; or create '.venv' inside your clone and install "
                "CosyVoice's requirements into it, then Re-check the engine."
            )
        return False, (
            "cosyvoice package not installed. Install it from Model Catalogue "
            "→ Engines."
        )

    @classmethod
    def venv_python(cls) -> Path:
        py = resolve_python()
        if py is None:
            raise RuntimeError(
                "CosyVoice's environment is missing or cannot import CosyVoice. "
                "Reinstall it from Model Catalogue → Engines."
            )
        return py

    @classmethod
    def sidecar_script(cls) -> Path:
        return Path(__file__).resolve().parent / "main.py"

    @property
    def recv_timeout_s(self) -> float:
        # Loading the model and its text normalizers takes a while on a cold
        # start; the sidecar heartbeats progress frames meanwhile.
        try:
            v = float(os.environ.get("OMNIVOICE_COSYVOICE_RECV_TIMEOUT_S", "900"))
        except (TypeError, ValueError):
            return 900.0
        if not math.isfinite(v):  # reject inf/nan so the deadline can't be disabled
            return 900.0
        return max(30.0, v)

    @property
    def sample_rate(self) -> int:
        # The sidecar resamples to this rate if a model ever reports another.
        return self._DEFAULT_SAMPLE_RATE

    @property
    def supported_languages(self) -> list[str]:
        from services.tts_backend import CosyVoiceBackend

        return CosyVoiceBackend.supported_languages.fget(self)

    def model_identity(self) -> str:
        # v1/v2/v3 share the one "cosyvoice" id; the folder name tells them
        # apart, as it does for the in-process engine.
        model_dir = os.environ.get("OMNIVOICE_COSYVOICE_MODEL") or MANAGED_MODEL_SUBDIR
        return os.path.basename(os.path.normpath(model_dir))


__all__ = [
    "CosyVoiceSubprocessBackend",
    "MANAGED_MODEL_SUBDIR",
    "VENV_ENV_VAR",
    "invalidate",
    "managed_python",
    "own_venv_python",
    "resolve_python",
    "user_venv_python",
]

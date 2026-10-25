from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def model_manager(monkeypatch):
    """Isolate model state and select native OmniVoice regardless of host hardware."""
    for mod_name in ("core.config", "services.model_manager"):
        if getattr(sys.modules.get(mod_name), "__file__", None) is None:
            sys.modules.pop(mod_name, None)

    import services.model_manager as mm
    from core import device_caps

    monkeypatch.setattr(
        device_caps, "detect_host_caps", lambda: SimpleNamespace(family="cpu")
    )
    monkeypatch.setenv("OMNIVOICE_TTS_BACKEND", "omnivoice")

    monkeypatch.setattr(mm, "_torch", None)
    monkeypatch.setattr(mm, "_OmniVoice", None)
    monkeypatch.setattr(mm, "model", None)
    monkeypatch.setenv("OMNIVOICE_MODEL", "test/checkpoint")
    return mm


def test_tts_asr_preload_is_opt_in(model_manager, monkeypatch):
    monkeypatch.delenv("OMNIVOICE_PRELOAD_TTS_ASR", raising=False)
    assert model_manager.should_preload_tts_asr() is False

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("OMNIVOICE_PRELOAD_TTS_ASR", value)
        assert model_manager.should_preload_tts_asr() is True

    monkeypatch.setenv("OMNIVOICE_PRELOAD_TTS_ASR", "0")
    assert model_manager.should_preload_tts_asr() is False


def test_load_model_skips_pytorch_whisper_by_default(model_manager, monkeypatch):
    calls = []

    class DummyOmniVoice:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(llm=object())

    monkeypatch.delenv("OMNIVOICE_PRELOAD_TTS_ASR", raising=False)
    monkeypatch.setattr(model_manager, "_lazy_torch", lambda: SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setattr(model_manager, "_lazy_omnivoice", lambda: DummyOmniVoice)
    monkeypatch.setattr(model_manager, "get_best_device", lambda: "mps")

    loaded = model_manager._load_model_sync()

    assert loaded.llm is not None
    assert loaded._voicestudio_checkpoint == "test/checkpoint"
    assert loaded._voicestudio_loaded_at.endswith("Z")
    assert calls == [
        (
            ("test/checkpoint",),
            {"device_map": "mps", "dtype": "float16", "load_asr": False},
        )
    ]


def test_load_model_can_preload_pytorch_whisper_when_requested(model_manager, monkeypatch):
    calls = []
    asr_loads = []

    class DummyModel:
        llm = object()

        def load_asr_model(self):
            asr_loads.append(True)

    class DummyOmniVoice:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return DummyModel()

    monkeypatch.setenv("OMNIVOICE_PRELOAD_TTS_ASR", "1")
    monkeypatch.setattr(model_manager, "_lazy_torch", lambda: SimpleNamespace(float16="float16", float32="float32"))
    monkeypatch.setattr(model_manager, "_lazy_omnivoice", lambda: DummyOmniVoice)
    monkeypatch.setattr(model_manager, "get_best_device", lambda: "mps")

    model_manager._load_model_sync()

    assert calls[0][1]["load_asr"] is False
    assert asr_loads == [True]


def test_resolve_checkpoint_honors_test_sentinel(model_manager, monkeypatch):
    """`OMNIVOICE_MODEL=test` (the suite-wide sentinel from tests/conftest.py)
    must pass through verbatim — the #693 bare-token self-heal mapping it to
    the real k2-fsa/OmniVoice checkpoint is what let app-booting tests kick
    off a real multi-GB model download."""
    monkeypatch.setenv("OMNIVOICE_MODEL", "test")
    assert model_manager.resolve_omnivoice_checkpoint() == "test"

    # The #693 self-heal itself must keep working for actual engine-id leaks.
    monkeypatch.setenv("OMNIVOICE_MODEL", "omnivoice")
    assert (
        model_manager.resolve_omnivoice_checkpoint()
        == model_manager._DEFAULT_OMNIVOICE_CHECKPOINT
    )


def test_preload_never_loads_uninstalled_checkpoint(model_manager, monkeypatch):
    """Networked machine, repo exists on the Hub, model NOT installed locally:
    preload must skip. The old `model_info()` probe treated Hub reachability
    as "installed" and silently downloaded the full checkpoint in a
    background thread on every boot with an empty cache."""
    import huggingface_hub

    monkeypatch.setenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    # Old-code determinism: make the Hub probe "succeed" without network.
    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **k: object(), raising=False
    )
    monkeypatch.setattr(model_manager, "_checkpoint_in_local_cache", lambda c: False)

    loads = []

    async def _record_load():
        loads.append(True)
        return SimpleNamespace(llm=object())

    monkeypatch.setattr(model_manager, "_load_model_with_timeout", _record_load)
    asyncio.run(model_manager.preload_model())

    assert loads == [], "preload must never load/download an uninstalled checkpoint"
    assert model_manager.model is None


@pytest.mark.parametrize("family", ["cpu", "cuda", "rocm"])
def test_preload_warms_up_locally_installed_checkpoint(model_manager, monkeypatch, family):
    """The counterpart guard: a locally present checkpoint must still warm up
    (skipping it would make the first /generate eat the full weight load)."""
    from core import device_caps

    monkeypatch.setattr(
        device_caps, "detect_host_caps", lambda: SimpleNamespace(family=family)
    )
    monkeypatch.setattr(model_manager, "_checkpoint_in_local_cache", lambda c: True)

    loaded = SimpleNamespace(llm=object())

    async def _fake_load():
        """Return installed weights without loading a real model."""
        return loaded

    monkeypatch.setattr(model_manager, "_load_model_with_timeout", _fake_load)
    asyncio.run(model_manager.preload_model())

    assert model_manager.model is loaded


@pytest.mark.parametrize("family", ["cpu", "cuda", "rocm", "mps"])
@pytest.mark.parametrize("backend_id", ["voxcpm2", "omnivoice-subprocess", "mlx-audio", "unknown"])
def test_preload_skips_other_selected_engines(model_manager, monkeypatch, family, backend_id):
    """Other engine selections must skip all native OmniVoice preload work."""
    from core import device_caps, prefs

    monkeypatch.setattr(
        device_caps, "detect_host_caps", lambda: SimpleNamespace(family=family)
    )
    monkeypatch.delenv("OMNIVOICE_TTS_BACKEND")
    prefs.set_("tts_backend", backend_id)

    def unexpected(*args, **kwargs):
        """Fail if the unselected engine reaches licence, cache, or weight work."""
        pytest.fail("an unselected OmniVoice must not check its licence, cache or load weights")

    monkeypatch.setattr(model_manager, "_ensure_omnivoice_licence", unexpected)
    monkeypatch.setattr(model_manager, "_checkpoint_in_local_cache", unexpected)
    monkeypatch.setattr(model_manager, "_load_model_with_timeout", unexpected)

    asyncio.run(model_manager.preload_model())

    assert model_manager.model is None


@pytest.mark.parametrize("saved,override,should_load", [
    ("omnivoice", "voxcpm2", False),
    ("voxcpm2", "omnivoice", True),
])
def test_preload_respects_engine_environment_override(
    model_manager, monkeypatch, saved, override, should_load
):
    """An environment override wins over the saved engine in either direction."""
    from core import prefs

    prefs.set_("tts_backend", saved)
    monkeypatch.setenv("OMNIVOICE_TTS_BACKEND", override)
    monkeypatch.setattr(model_manager, "_checkpoint_in_local_cache", lambda c: True)
    loaded = SimpleNamespace(llm=object())

    async def fake_load():
        """Return a recognizable model to detect whether preload ran."""
        return loaded

    monkeypatch.setattr(model_manager, "_load_model_with_timeout", fake_load)
    asyncio.run(model_manager.preload_model())

    assert model_manager.model is (loaded if should_load else None)


def test_preload_keeps_default_omnivoice_warmup(model_manager, monkeypatch):
    """Missing engine preferences and overrides preserve the default warm-up."""
    from core import prefs

    monkeypatch.delenv("OMNIVOICE_TTS_BACKEND")
    prefs.delete("tts_backend")
    monkeypatch.setattr(model_manager, "_checkpoint_in_local_cache", lambda c: True)
    loaded = SimpleNamespace(llm=object())

    async def fake_load():
        """Return a recognizable model without downloading or allocating weights."""
        return loaded

    monkeypatch.setattr(model_manager, "_load_model_with_timeout", fake_load)
    asyncio.run(model_manager.preload_model())

    assert model_manager.model is loaded

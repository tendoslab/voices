"""Locking a take keeps its whole text as the profile's reference transcript.

The lock used to cut the take's text to 100 characters. Generation pairs the
locked take with that text, so a take longer than that was cloned against a
transcript that stopped partway through the audio (#3254).
"""
from __future__ import annotations

import io
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TAKE_TEXT = (
    "Ontem à noite o vento soprou forte sobre a cidade, e as janelas da velha "
    "casa tremeram até o amanhecer, quando finalmente tudo ficou em silêncio."
)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\x00\x01" * 64)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    from api.routers import profiles
    from core import db

    voices, outputs = tmp_path / "voices", tmp_path / "outputs"
    voices.mkdir()
    outputs.mkdir()
    (outputs / "take.wav").write_bytes(_wav_bytes())
    monkeypatch.setattr(profiles, "VOICES_DIR", str(voices))
    monkeypatch.setattr(profiles, "OUTPUTS_DIR", str(outputs))
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "profiles.db"))
    db.init_db()
    with db.db_conn() as conn:
        conn.execute(
            "INSERT INTO voice_profiles (id, name, ref_audio_path, ref_text) "
            "VALUES ('p1', 'Narrator', '', 'original reference')"
        )
        conn.execute(
            "INSERT INTO generation_history (id, text, audio_path) VALUES ('h1', ?, 'take.wav')",
            (TAKE_TEXT,),
        )
        conn.commit()
    app = FastAPI()
    app.include_router(profiles.router)
    with TestClient(app) as test_client:
        yield test_client


def test_lock_keeps_the_whole_take_text(client):
    from core import db

    assert len(TAKE_TEXT) > 100
    response = client.post("/profiles/p1/lock", data={"history_id": "h1", "seed": "7"})
    assert response.status_code == 200, response.text
    with db.db_conn() as conn:
        row = conn.execute("SELECT ref_text, seed, is_locked FROM voice_profiles WHERE id='p1'").fetchone()
    assert row["ref_text"] == TAKE_TEXT
    assert (row["seed"], row["is_locked"]) == (7, 1)

"""custom-voice: STT + TTS for Hermes with built-in multi-key rotation.

Two modes, picked automatically:

1. **Direct keys (preferred)** — ``CUSTOM_VOICE_KEYS_FILE``
   (default ``~/.hermes/custom-voice-keys.yaml``). The plugin calls Google
   (Gemini TTS) and Groq (Whisper STT) directly and round-robins through all
   your keys, cooling down keys that hit 429/5xx. No gateway required.

2. **Gateway** — ``CUSTOM_VOICE_BASE_URL``: any OpenAI-compatible speech
   endpoint (e.g. a 9Router LAN instance). Used when no keys file exists.

Keys file format::

    gemini:
      tts_model: gemini-3.8-flash-lite-tts
      tts_voice: Autonoe
      keys:
        - name: me
          key: AQ...
    groq:
      stt_model: whisper-large-v3
      keys:
        - name: me
          key: gsk_...
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

__all__ = ["register"]

NAME = "custom"
_KEYS_FILE_DEFAULT = "~/.hermes/custom-voice-keys.yaml"
_GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
_GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_COOLDOWN_SECONDS = 900  # 15 min after a 429/quota strike
_CTX: Any = None


def _get_setting(key: str, default: Any = None) -> Any:
    if _CTX is not None and hasattr(_CTX, "get_config"):
        try:
            val = _CTX.get_config(key, default=None)
            if val is not None and str(val).strip():
                return val
        except Exception:
            pass
    return default

# ---------------------------------------------------------------------------
# Key store + rotation
# ---------------------------------------------------------------------------


class _KeyRing:
    """Round-robin key ring with per-key cooldown after failures."""

    def __init__(self) -> None:
        self._keys: List[Dict[str, str]] = []
        self._idx = 0
        self._cooldown_until: Dict[int, float] = {}

    def load(self, keys: List[Dict[str, str]]) -> None:
        self._keys = [k for k in keys if isinstance(k, dict) and k.get("key")]
        self._idx = 0
        self._cooldown_until = {}

    def __bool__(self) -> bool:
        return bool(self._keys)

    def order(self) -> List[int]:
        """Indices in try-order: non-cooled first (round-robin start), then cooled."""
        n = len(self._keys)
        if not n:
            return []
        now = time.time()
        hot = [i for i in range(n) if self._cooldown_until.get(i, 0) <= now]
        cold = [i for i in range(n) if self._cooldown_until.get(i, 0) > now]
        if hot:
            hot = hot[self._idx % len(hot):] + hot[: self._idx % len(hot)]
        return hot + cold

    def pick(self, i: int) -> str:
        return self._keys[i]["key"]

    def mark_ok(self, i: int) -> None:
        self._idx = (i + 1) % max(1, len(self._keys))
        self._cooldown_until.pop(i, None)

    def mark_failed(self, i: int) -> None:
        self._cooldown_until[i] = time.time() + _COOLDOWN_SECONDS


def _keys_file() -> Optional[Path]:
    raw = _get_setting("keys_file") or os.getenv("CUSTOM_VOICE_KEYS_FILE", _KEYS_FILE_DEFAULT)
    p = Path(os.path.expanduser(str(raw)))
    return p if p.is_file() else None


def _load_keys() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (gemini_cfg, groq_cfg) from the keys file; ({}, {}) when absent."""
    p = _keys_file()
    if not p:
        return {}, {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text()) or {}
        return data.get("gemini") or {}, data.get("groq") or {}
    except Exception as exc:  # never crash provider registration
        logger.warning("custom-voice: cannot read keys file %s: %s", p, exc)
        return {}, {}


_GEMINI_RING = _KeyRing()
_GROQ_RING = _KeyRing()
_GEMINI_CFG: Dict[str, Any] = {}
_GROQ_CFG: Dict[str, Any] = {}


def _init_rings() -> None:
    global _GEMINI_CFG, _GROQ_CFG
    gem, groq = _load_keys()
    _GEMINI_CFG, _GROQ_CFG = gem, groq
    _GEMINI_RING.load(gem.get("keys") or [])
    _GROQ_RING.load(groq.get("keys") or [])


def _base_url() -> str:
    url = _get_setting("gateway_url") or os.getenv("CUSTOM_VOICE_BASE_URL", "")
    return str(url).rstrip("/")


def _gateway_key() -> str:
    return os.getenv("CUSTOM_VOICE_API_KEY", "not-needed")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

_GEMINI_DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"
_GEMINI_DEFAULT_VOICE = "Autonoe"


def _gemini_tts_once(key: str, model: str, voice: str, text: str) -> bytes:
    """One Gemini TTS call -> raw audio bytes (wav container or L16 PCM)."""
    import requests

    body = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
            },
        },
    }
    r = requests.post(
        _GEMINI_TTS_URL.format(model=model, key=key), json=body, timeout=90
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    cands = data.get("candidates") or []
    if not cands:
        raise RuntimeError(f"no candidates: {str(data)[:200]}")
    for part in cands[0].get("content", {}).get("parts", []):
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])
    raise RuntimeError("no audio part in response")


def _pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_to_mp3(wav_bytes: bytes, out_path: str) -> bool:
    """Convert with Hermes' bundled ffmpeg. True on success."""
    import shutil
    import tempfile

    candidates = [
        os.path.expanduser("~/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"),
    ]
    ff = shutil.which("ffmpeg") or next(
        (c for c in candidates if os.path.exists(c)), None
    )
    if not ff:
        return False
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        res = subprocess.run(
            [ff, "-y", "-i", tmp_path, "-codec:a", "libmp3lame", "-qscale:a", "4", out_path],
            capture_output=True,
            timeout=60,
        )
        return res.returncode == 0 and os.path.exists(out_path)
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class CustomTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return NAME

    @property
    def display_name(self) -> str:
        return "Custom Voice (TTS)"

    @property
    def voice_compatible(self) -> bool:
        from gateway.session_context import get_session_env
        from tools.tts_tool import OPUS_VOICE_PLATFORMS

        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        return str(platform or "").strip().lower() in OPUS_VOICE_PLATFORMS

    def is_available(self) -> bool:
        return bool(_GEMINI_RING) or bool(_base_url())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "TTS with your own keys (rotating) or endpoint",
            "env_vars": [
                {"key": "CUSTOM_VOICE_KEYS_FILE", "prompt": "YAML keys file (direct mode)"},
                {"key": "CUSTOM_VOICE_BASE_URL", "prompt": "OpenAI-compatible base URL (gateway mode)"},
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": "gemini-3.8-flash-lite-tts", "display": "gemini-3.8-flash-lite-tts (Best Value / Low Price)"},
            {"id": "gemini-3.8-flash-tts", "display": "gemini-3.8-flash-tts (Highest Quality)"},
            {"id": "gemini-2.5-flash-preview-tts", "display": "gemini-2.5-flash-preview-tts"},
            {"id": "gemini-2.5-pro-preview-tts", "display": "gemini-2.5-pro-preview-tts"},
        ]

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = [
            "Autonoe", "Kore", "Puck", "Zephyr", "Fenrir", "Aoede", "Leda", "Callirrhoe",
            "Charon", "Orus", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome"
        ]
        return [{"id": v, "display": v} for v in voices]

    def default_model(self) -> Optional[str]:
        return str(_get_setting("tts_model") or _GEMINI_CFG.get("tts_model") or _GEMINI_DEFAULT_MODEL)

    def default_voice(self) -> Optional[str]:
        return str(_get_setting("tts_voice") or _GEMINI_CFG.get("tts_voice") or _GEMINI_DEFAULT_VOICE)

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        want_mp3 = str(format or "mp3").lower() in {"mp3", "opus", "ogg", "aac"}

        if _GEMINI_RING:
            return self._synthesize_direct(text, target, voice, model, want_mp3)
        if _base_url():
            return self._synthesize_gateway(text, target, voice, model, speed, format)
        raise RuntimeError("custom TTS: no keys file and no CUSTOM_VOICE_BASE_URL")

    def _synthesize_direct(
        self, text: str, target: Path, voice: Optional[str], model: Optional[str], want_mp3: bool
    ) -> str:
        last_err: Optional[Exception] = None
        sel_model = str(model or _get_setting("tts_model") or _GEMINI_CFG.get("tts_model") or _GEMINI_DEFAULT_MODEL)
        sel_voice = str(voice or _get_setting("tts_voice") or _GEMINI_CFG.get("tts_voice") or _GEMINI_DEFAULT_VOICE)
        for i in _GEMINI_RING.order():
            try:
                audio = _gemini_tts_once(_GEMINI_RING.pick(i), sel_model, sel_voice, text)
                _GEMINI_RING.mark_ok(i)
                wav = audio if audio[:4] == b"RIFF" else _pcm_to_wav(audio)
                wrote = _wav_to_mp3(wav, str(target)) if want_mp3 else False
                if not wrote:
                    target.write_bytes(wav)
                return str(target)
            except Exception as exc:
                last_err = exc
                _GEMINI_RING.mark_failed(i)
                logger.warning(
                    "custom-voice TTS key #%d failed (%s), rotating", i, str(exc)[:120]
                )
        raise RuntimeError(f"custom TTS: all keys failed: {last_err}")

    def _synthesize_gateway(
        self,
        text: str,
        target: Path,
        voice: Optional[str],
        model: Optional[str],
        speed: Optional[float],
        format: str,
    ) -> str:
        from openai import OpenAI

        response_format = str(format or "mp3").lower()
        if response_format == "ogg":
            response_format = "opus"
        if response_format not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            response_format = "mp3"
        client = OpenAI(api_key=_gateway_key(), base_url=_base_url(), timeout=90, max_retries=0)
        try:
            kwargs: Dict[str, Any] = {
                "model": model or os.getenv("CUSTOM_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
                "voice": voice or os.getenv("CUSTOM_TTS_VOICE", "Autonoe"),
                "input": text,
                "response_format": response_format,
            }
            if speed is not None:
                kwargs["speed"] = max(0.25, min(4.0, float(speed)))
            response = client.audio.speech.create(**kwargs)
            response.stream_to_file(target)
            return str(target)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


def _groq_stt_once(key: str, model: str, file_path: str, language: Optional[str]) -> str:
    """One Groq Whisper call -> transcript text."""
    import requests

    data: Dict[str, str] = {"model": model, "response_format": "json"}
    if language:
        data["language"] = language
    with open(file_path, "rb") as fh:
        r = requests.post(
            _GROQ_STT_URL,
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files={"file": (os.path.basename(file_path), fh)},
            timeout=120,
        )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    body = r.json()
    if isinstance(body, str):
        return body
    text = str(body.get("text") or "")
    if not text:
        raise RuntimeError(f"empty transcript: {str(body)[:200]}")
    return text


class CustomTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return NAME

    @property
    def display_name(self) -> str:
        return "Custom Voice (STT)"

    def is_available(self) -> bool:
        return bool(_GROQ_RING) or bool(_base_url())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "STT with your own keys (rotating) or endpoint",
            "env_vars": [
                {"key": "CUSTOM_VOICE_KEYS_FILE", "prompt": "YAML keys file (direct mode)"},
                {"key": "CUSTOM_VOICE_BASE_URL", "prompt": "OpenAI-compatible base URL (gateway mode)"},
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": "whisper-large-v3", "display": "whisper-large-v3 (Default / High Accuracy)"},
            {"id": "whisper-large-v3-turbo", "display": "whisper-large-v3-turbo (Fast)"},
        ]

    def default_model(self) -> Optional[str]:
        return str(_get_setting("stt_model") or _GROQ_CFG.get("stt_model") or "whisper-large-v3")

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        if _GROQ_RING:
            last_err: Optional[Exception] = None
            sel_model = str(model or _get_setting("stt_model") or _GROQ_CFG.get("stt_model") or "whisper-large-v3")
            for i in _GROQ_RING.order():
                try:
                    text = _groq_stt_once(_GROQ_RING.pick(i), sel_model, file_path, language)
                    _GROQ_RING.mark_ok(i)
                    return {"success": True, "transcript": text, "provider": self.name}
                except Exception as exc:
                    last_err = exc
                    _GROQ_RING.mark_failed(i)
                    logger.warning(
                        "custom-voice STT key #%d failed (%s), rotating", i, str(exc)[:120]
                    )
            return {
                "success": False,
                "transcript": "",
                "error": f"custom transcription: all keys failed: {last_err}",
                "provider": self.name,
            }
        if _base_url():
            return self._transcribe_gateway(file_path, model, language)
        return {
            "success": False,
            "transcript": "",
            "error": "custom STT: no keys file and no CUSTOM_VOICE_BASE_URL",
            "provider": self.name,
        }

    def _transcribe_gateway(
        self, file_path: str, model: Optional[str], language: Optional[str]
    ) -> Dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(api_key=_gateway_key(), base_url=_base_url(), timeout=120, max_retries=0)
        try:
            kwargs: Dict[str, Any] = {
                "model": model or os.getenv("CUSTOM_STT_MODEL", "groq/whisper-large-v3"),
                "response_format": "json",
            }
            if language:
                kwargs["language"] = language
            with open(file_path, "rb") as audio_file:
                kwargs["file"] = audio_file
                response = client.audio.transcriptions.create(**kwargs)
            if isinstance(response, str):
                transcript = response
            elif isinstance(response, dict):
                transcript = str(response.get("text") or "")
            else:
                transcript = str(getattr(response, "text", "") or "")
            return {"success": True, "transcript": transcript, "provider": self.name}
        except Exception as exc:
            return {
                "success": False,
                "transcript": "",
                "error": f"custom transcription failed: {str(exc)[:300]}",
                "provider": self.name,
            }
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    global _CTX
    _CTX = ctx
    _init_rings()
    ctx.register_tts_provider(CustomTTSProvider())
    ctx.register_transcription_provider(CustomTranscriptionProvider())
    mode = "direct-keys" if (_GEMINI_RING or _GROQ_RING) else (
        f"gateway({_base_url()})" if _base_url() else "unconfigured"
    )
    logger.info(
        "custom-voice registered (mode=%s, gemini_keys=%d, groq_keys=%d)",
        mode,
        len(_GEMINI_RING._keys),
        len(_GROQ_RING._keys),
    )

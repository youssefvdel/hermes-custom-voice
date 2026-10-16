"""Custom-domain STT + TTS via any OpenAI-compatible speech endpoint.

Env:
  CUSTOM_VOICE_BASE_URL  e.g. http://192.168.100.5:20128/v1 (9Router default)
  CUSTOM_VOICE_API_KEY   bearer key (9Router LAN endpoint may need none)
  CUSTOM_STT_MODEL       default: groq/whisper-large-v3-turbo
  CUSTOM_TTS_MODEL       default: gemini-2.5-flash-preview-tts (override for Gemini via 9Router)
  CUSTOM_TTS_VOICE       default: alloy
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

__all__ = ["register"]

NAME = "custom"


def _base_url() -> str:
    return os.getenv("CUSTOM_VOICE_BASE_URL", "http://192.168.100.5:20128/v1").rstrip("/")


def _api_key() -> str:
    return os.getenv("CUSTOM_VOICE_API_KEY", "not-needed")


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
        return bool(_base_url())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "TTS through your own endpoint",
            "env_vars": [
                {"key": "CUSTOM_VOICE_BASE_URL", "prompt": "OpenAI-compatible base URL"},
                {"key": "CUSTOM_VOICE_API_KEY", "prompt": "API key (may be unneeded on LAN)"},
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": os.getenv("CUSTOM_TTS_MODEL", "gemini-2.5-flash-preview-tts"), "display": "custom"}]

    def list_voices(self) -> List[Dict[str, Any]]:
        return [{"id": os.getenv("CUSTOM_TTS_VOICE", "Kore"), "display": "custom"}]

    def default_model(self) -> Optional[str]:
        return os.getenv("CUSTOM_TTS_MODEL", "gemini-2.5-flash-preview-tts")

    def default_voice(self) -> Optional[str]:
        return os.getenv("CUSTOM_TTS_VOICE", "Kore")

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
        from openai import OpenAI

        response_format = str(format or "mp3").lower()
        if response_format == "ogg":
            response_format = "opus"
        if response_format not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            response_format = "mp3"
        client = OpenAI(api_key=_api_key(), base_url=_base_url(), timeout=60, max_retries=0)
        try:
            kwargs: Dict[str, Any] = {
                "model": model or self.default_model(),
                "voice": voice or self.default_voice(),
                "input": text,
                "response_format": response_format,
            }
            if speed is not None:
                kwargs["speed"] = max(0.25, min(4.0, float(speed)))
            response = client.audio.speech.create(**kwargs)
            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            response.stream_to_file(target)
            return str(target)
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()


class CustomTranscriptionProvider(TranscriptionProvider):
    @property
    def name(self) -> str:
        return NAME

    @property
    def display_name(self) -> str:
        return "Custom Voice (STT)"

    def is_available(self) -> bool:
        return bool(_base_url())

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "custom",
            "tag": "STT through your own endpoint",
            "env_vars": [
                {"key": "CUSTOM_VOICE_BASE_URL", "prompt": "OpenAI-compatible base URL"},
                {"key": "CUSTOM_VOICE_API_KEY", "prompt": "API key (may be unneeded on LAN)"},
            ],
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": os.getenv("CUSTOM_STT_MODEL", "groq/whisper-large-v3-turbo"), "display": "custom"}]

    def default_model(self) -> Optional[str]:
        return os.getenv("CUSTOM_STT_MODEL", "groq/whisper-large-v3-turbo")

    def transcribe(
        self,
        file_path: str,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        from openai import OpenAI

        client = OpenAI(api_key=_api_key(), base_url=_base_url(), timeout=60, max_retries=0)
        try:
            kwargs: Dict[str, Any] = {
                "model": model or self.default_model(),
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


def register(ctx) -> None:
    ctx.register_tts_provider(CustomTTSProvider())
    ctx.register_transcription_provider(CustomTranscriptionProvider())
    logger.info("custom-voice providers registered (base=%s)", _base_url())

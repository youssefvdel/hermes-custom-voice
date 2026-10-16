# hermes-custom-voice

STT + TTS for Hermes Agent through **any OpenAI-compatible speech endpoint** — built for [9Router](https://github.com/decolua/9router), works with anything speaking `/v1/audio/*`.

## Why

Hermes hardcodes its voice providers. This plugin registers a `custom` transcription +
TTS provider pair that points at whatever base URL you set, so all your pooled
provider keys (Groq STT, Gemini TTS, …) route through one gateway.

## Install

```bash
cp -r custom-voice ~/.hermes/plugins/
hermes plugins enable custom-voice
```

## Configure

```bash
CUSTOM_VOICE_BASE_URL=http://192.168.100.5:20128/v1   # your gateway
CUSTOM_VOICE_API_KEY=sk-...                            # its API key
CUSTOM_STT_MODEL=groq/whisper-large-v3-turbo
CUSTOM_TTS_MODEL=gemini-2.5-flash-preview-tts
CUSTOM_TTS_VOICE=Kore
```

Then:

```bash
hermes config set tts.provider custom
hermes config set stt.provider custom   # if your build reads it, else autodetect
systemctl --user restart hermes-serve.service  # so the backend picks up env
```

## Verified live

- STT: whisper via 9Router → exact transcript, pooled Groq keys, round-robin.
- TTS: Gemini TTS via 9Router → real audio (WAV/MP3), pooled Gemini keys.

## Layout

- `plugin.yaml` — manifest (no tools, no hooks, providers only)
- `__init__.py` — `register(ctx)` + both provider classes, nothing else

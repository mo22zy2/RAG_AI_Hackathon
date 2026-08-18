import logging
import edge_tts

from . import TTSInterface

logger = logging.getLogger(__name__)


class EdgeTTSProvider(TTSInterface):
    """Microsoft Edge TTS provider — free, no API key, async-native."""

    DEFAULT_VOICE = "ar-EG-SalmaNeural"

    def __init__(self, default_voice: str = None):
        self.default_voice = default_voice or self.DEFAULT_VOICE
        self._voices_cache = None

    async def synthesize(self, text: str, voice: str = None) -> bytes:
        voice = voice or self.default_voice
        text = (text or "").strip()
        if not text:
            return b""

        try:
            communicate = edge_tts.Communicate(text, voice)
            audio = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return audio
        except Exception as e:
            logger.error("TTS synthesis failed for voice=%s: %s", voice, e)
            raise

    async def list_voices(self, language: str = None) -> list:
        if self._voices_cache is None:
            self._voices_cache = await edge_tts.list_voices()

        if not language:
            return [v["ShortName"] for v in self._voices_cache]

        lang = language.lower()
        return [
            v["ShortName"]
            for v in self._voices_cache
            if v.get("Locale", "").lower().startswith(lang)
        ]

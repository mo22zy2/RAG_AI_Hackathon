from abc import ABC, abstractmethod


class TTSInterface(ABC):
    """Abstract interface for text-to-speech providers."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Convert text to audio bytes (MP3 format).

        Args:
            text: The text to synthesize.
            voice: Optional voice identifier. If None, uses the provider default.

        Returns:
            Raw MP3 audio bytes.
        """
        pass

    @abstractmethod
    def list_voices(self, language: str = None) -> list:
        """Return available voices, optionally filtered by language prefix.

        Args:
            language: Language prefix to filter by (e.g. "ar" for Arabic).

        Returns:
            List of voice identifiers.
        """
        pass

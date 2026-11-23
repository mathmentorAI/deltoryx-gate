"""Internal source contracts and provider transport used by the Gate layers."""
from .models import SourceDocument, EvidenceRef
from .adapters import OpenAICompatibleModel

__all__ = ["SourceDocument", "EvidenceRef", "OpenAICompatibleModel"]

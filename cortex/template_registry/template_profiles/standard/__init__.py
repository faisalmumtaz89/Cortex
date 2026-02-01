"""Standard template profiles."""

from cortex.template_registry.template_profiles.standard.chatml import ChatMLProfile
from cortex.template_registry.template_profiles.standard.llama import LlamaProfile
from cortex.template_registry.template_profiles.standard.alpaca import AlpacaProfile
from cortex.template_registry.template_profiles.standard.simple import SimpleProfile
from cortex.template_registry.template_profiles.standard.gemma import GemmaProfile

__all__ = ['ChatMLProfile', 'LlamaProfile', 'AlpacaProfile', 'SimpleProfile', 'GemmaProfile']